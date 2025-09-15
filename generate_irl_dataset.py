import aisdb
from aisdb.database.dbconn import PostgresDBConn
from aisdb.database.dbqry import DBQuery
from aisdb.database import sqlfcn_callbacks, sqlfcn
from aisdb.discretize.h3 import Discretizer
from aisdb.ports.api import WorldPortIndexClient
from aisdb.denoising_encoder import encode_greatcircledistance
import aisdb.track_gen
import aisdb.interp

from datetime import datetime, timedelta
import pandas as pd
import os

# -------------------------------
# 1. PostgreSQL Connection Setup
# -------------------------------
db_user = "vaishnav"
db_dbname = "aisviz"
db_password = "<>"
db_hostaddr = "127.0.0.1"
db_port = 5432

xmin, ymin, xmax, ymax = -70, 45, -58, 53
time_split = timedelta(hours=3)
distance_split = 10000  # meters
speed_split = 40  # knots

start_time = datetime(2024, 1, 1, 0, 0, 0)
end_time = datetime(2024, 1, 5, 23, 59, 59)

# -------------------------------
# 2. World Port Index (reference data)
# -------------------------------
client = WorldPortIndexClient()
df_ports = client.fetch_ports(lat_min=ymin, lat_max=ymax, lon_min=xmin, lon_max=xmax)

# Discretizer
discretizer = Discretizer(resolution=5)

# Cast port coordinates
df_ports = df_ports.rename(columns={"LAT": "lat", "LON": "lon"})
df_ports[["lat", "lon"]] = df_ports[["lat", "lon"]].astype(float)

# Assign H3 index
df_ports["h3_index"] = df_ports.apply(
    lambda r: discretizer.get_h3_index(r["lat"], r["lon"]), axis=1
)

# -------------------------------
# 3. Query AISDB trajectories
# -------------------------------
with PostgresDBConn(
    host=db_hostaddr,
    port=db_port,
    user=db_user,
    dbname=db_dbname,
    password=db_password,
) as dbconn:
    print(f"Connected to {db_dbname}")

    qry = DBQuery(
        dbconn=dbconn,
        start=start_time,
        end=end_time,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        callback=aisdb.database.sqlfcn_callbacks.in_time_bbox_validmmsi,
    )

    rowgen = qry.gen_qry(fcn=sqlfcn.crawl_dynamic_static)
    tracks_raw = aisdb.track_gen.TrackGen(rowgen, decimate=True)

    track_segments = aisdb.track_gen.split_timedelta(tracks_raw, time_split)
    tracks_encoded = encode_greatcircledistance(
        track_segments,
        distance_threshold=distance_split,
        speed_threshold=speed_split,
    )
    tracks = aisdb.interp.interp_time(tracks_encoded, step=timedelta(minutes=1))
    tracks_discretized = discretizer.yield_tracks_discretized_by_indexes(tracks)

    # -------------------------------
    # 4. Stop detection
    # -------------------------------
    SPEED_THRESHOLD_KNOTS = 0.5
    MIN_STOP_DURATION_MIN = 30

    stops = []
    for track in tracks_discretized:
        mmsi = track["mmsi"]
        ship_type = track.get("ship_type_txt") or "Unknown"
        times, sogs, h3s = track["time"], track["sog"], track["h3_index"]

        current = []
        for i, sog in enumerate(sogs):
            if sog <= SPEED_THRESHOLD_KNOTS:
                current.append((times[i], h3s[i]))
            else:
                if current:
                    first_t, last_t = current[0][0], current[-1][0]
                    dur = (int(last_t) - int(first_t)) / 60
                    if dur >= MIN_STOP_DURATION_MIN:
                        stops.append(
                            {
                                "mmsi": mmsi,
                                "ship_type": ship_type,
                                "start_time": int(first_t),
                                "end_time": int(last_t),
                                "duration_min": dur,
                                "h3_index": current[0][1],
                            }
                        )
                    current = []

        if current:  # end-of-track stop
            first_t, last_t = current[0][0], current[-1][0]
            dur = (int(last_t) - int(first_t)) / 60
            if dur >= MIN_STOP_DURATION_MIN:
                stops.append(
                    {
                        "mmsi": mmsi,
                        "ship_type": ship_type,
                        "start_time": int(first_t),
                        "end_time": int(last_t),
                        "duration_min": dur,
                        "h3_index": current[0][1],
                    }
                )

    stops_df = pd.DataFrame(stops)
    if stops_df.empty:
        raise ValueError("No stops detected in this query region/time.")

    # -------------------------------
    # 5. Match stops to ports
    # -------------------------------
    stops_df = stops_df.merge(
        df_ports[["PORT_NAME", "h3_index"]], on="h3_index", how="left"
    )
    stops_df = stops_df.dropna(subset=["PORT_NAME"])
    stops_df = stops_df.sort_values(["mmsi", "start_time"]).reset_index(drop=True)

    # -------------------------------
    # 6. Build collapsed port & H3 sequences
    # -------------------------------
    def collapse_consecutive(seq_list):
        if not seq_list:
            return []
        collapsed = [seq_list[0]]
        for item in seq_list[1:]:
            if item != collapsed[-1]:
                collapsed.append(item)
        return collapsed

    def build_port_sequence(group):
        ports = group["PORT_NAME"].tolist()
        collapsed_ports = collapse_consecutive(ports)
        return ",".join(collapsed_ports)

    def build_h3_sequence(group):
        h3s = group["h3_index"].tolist()
        collapsed_h3s = collapse_consecutive(h3s)
        return ",".join(collapsed_h3s)

    port_seq_df = (
        stops_df.groupby("mmsi")
        .apply(build_port_sequence)
        .reset_index(name="PORT_Sequence")
    )
    h3_seq_df = (
        stops_df.groupby("mmsi")
        .apply(build_h3_sequence)
        .reset_index(name="H3_Sequence")
    )

    stops_with_seq = stops_df.merge(port_seq_df, on="mmsi", how="left")
    stops_with_seq = stops_with_seq.merge(h3_seq_df, on="mmsi", how="left")

    stops_with_seq["start_time"] = pd.to_datetime(
        stops_with_seq["start_time"], unit="s"
    )
    stops_with_seq["end_time"] = pd.to_datetime(stops_with_seq["end_time"], unit="s")

    final_df = stops_with_seq[
        [
            "mmsi",
            "ship_type",
            "start_time",
            "end_time",
            "duration_min",
            "h3_index",
            "PORT_Sequence",
            "H3_Sequence",
        ]
    ]

    print("\n--- Final Collapsed Port Visits with Sequences ---")
    print(final_df.head())

    # -------------------------------
    # 7. Save separate CSVs per ship_type
    # -------------------------------
    year = start_time.year
    output_dir = f"output/{year}"
    os.makedirs(output_dir, exist_ok=True)

    for ship_type, group in final_df.groupby("ship_type"):
        safe_type = ship_type.replace("/", "_").replace(" ", "_")
        out_path = os.path.join(output_dir, f"{safe_type}.csv")
        group.to_csv(out_path, index=False)
        print(f"Saved {len(group)} rows for {ship_type} -> {out_path}")
