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

# Spatial and temporal filtering
xmin, ymin, xmax, ymax = -70, 44.9585, -58, 52.2224
time_split = timedelta(hours=3)
distance_split = 10000  # meters
speed_split = 40  # knots
start_time = datetime(2024, 1, 1, 0, 0, 0)
end_time = datetime(2024, 4, 3, 23, 59, 59)

# -------------------------------
# 2. Load World Port Index as reference data
# -------------------------------
client = WorldPortIndexClient()
df_ports = client.fetch_ports(lat_min=45.0, lat_max=51.5, lon_min=-71.5, lon_max=-55.0)

# Initialize H3 discretizer
discretizer = Discretizer(resolution=5)

# Normalize coordinate columns
df_ports = df_ports.rename(columns={"LAT": "lat", "LON": "lon"})
df_ports[["lat", "lon"]] = df_ports[["lat", "lon"]].astype(float)

# Assign H3 spatial index to each port
df_ports["h3_index"] = df_ports.apply(
    lambda r: discretizer.get_h3_index(r["lat"], r["lon"]), axis=1
)

# -------------------------------
# 3. Query AIS vessel trajectories
# -------------------------------
with PostgresDBConn(
    host=db_hostaddr,
    port=db_port,
    user=db_user,
    dbname=db_dbname,
    password=db_password,
) as dbconn:
    print(f"Connected to {db_dbname}")

    # Query vessel positions within bounding box & time range
    qry = DBQuery(
        dbconn=dbconn,
        start=start_time,
        end=end_time,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        callback=sqlfcn_callbacks.in_time_bbox_validmmsi,
    )

    # Get dynamic/static vessel data
    rowgen = qry.gen_qry(fcn=sqlfcn.crawl_dynamic_static)
    tracks_raw = aisdb.track_gen.TrackGen(rowgen, decimate=True)

    # Split tracks into segments by time gap
    track_segments = aisdb.track_gen.split_timedelta(tracks_raw, time_split)

    # Encode great-circle distances and remove unrealistic jumps
    tracks_encoded = encode_greatcircledistance(
        track_segments, distance_threshold=distance_split, speed_threshold=speed_split
    )

    # Interpolate tracks with 1-minute granularity
    tracks = aisdb.interp.interp_time(tracks_encoded, step=timedelta(minutes=1))

    # Discretize vessel positions to H3 cells
    tracks_discretized = discretizer.yield_tracks_discretized_by_indexes(tracks)

    # -------------------------------
    # 4. Stop detection (anchorage/port stay inference)
    # -------------------------------
    SPEED_THRESHOLD_KNOTS = 0.5  # vessel considered "stopped"
    MIN_STOP_DURATION_MIN = 30  # minimum stop duration for port visit

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
                # If stop found, calculate stop duration
                if current:
                    first_t, last_t = current[0][0], current[-1][0]
                    dur = (int(last_t) - int(first_t)) / 60  # seconds → minutes
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
        # Handle trailing stop sequences at end of track
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

    stops_df = pd.DataFrame(stops)
    if stops_df.empty:
        raise ValueError("No stops detected in this query region/time.")

    # -------------------------------
    # 5. Match detected stops to ports
    # -------------------------------
    stops_df = stops_df.merge(
        df_ports[["PORT_NAME", "h3_index"]], on="h3_index", how="left"
    )

    # Filter out stops not at known ports
    stops_df = stops_df.dropna(subset=["PORT_NAME"])

    # Chronological ordering
    stops_df = stops_df.sort_values(["mmsi", "start_time"]).reset_index(drop=True)

    # -------------------------------
    # 6. Build collapsed sequences (port visit chains)
    # -------------------------------
    def collapse_consecutive(seq_list):
        """Collapse consecutive duplicate entries in sequence list."""
        if not seq_list:
            return []
        collapsed = [seq_list[0]]
        for item in seq_list[1:]:
            if item != collapsed[-1]:
                collapsed.append(item)
        return collapsed

    def build_port_sequence(group):
        ports = group["PORT_NAME"].tolist()
        return ",".join(collapse_consecutive(ports))

    def build_h3_sequence(group):
        h3s = group["h3_index"].tolist()
        return ",".join(collapse_consecutive(h3s))

    # Generate collapsed sequences for each vessel
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

    # Merge sequences back into stop-level table
    stops_with_seq = stops_df.merge(port_seq_df, on="mmsi", how="left")
    stops_with_seq = stops_with_seq.merge(h3_seq_df, on="mmsi", how="left")

    # Convert timestamps to human-readable datetime
    stops_with_seq["start_time"] = pd.to_datetime(
        stops_with_seq["start_time"], unit="s"
    )
    stops_with_seq["end_time"] = pd.to_datetime(stops_with_seq["end_time"], unit="s")

    # Select final columns
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

    # Save global results
    final_df.to_csv("port_visits_with_sequences_collapsed.csv", index=False)

    # -------------------------------
    # 7. Save Separate Results by Year & Ship Type
    # -------------------------------
    ship_type_csvs = {}
    year = start_time.year

    base_dir = os.path.join("output", str(year))
    os.makedirs(base_dir, exist_ok=True)

    # Write per-ship-type CSV files
    for ship_type, group_df in final_df.groupby("ship_type"):
        safe_ship_type = ship_type.replace(" ", "_").replace("/", "-")
        ship_dir = os.path.join(base_dir, safe_ship_type)
        os.makedirs(ship_dir, exist_ok=True)

        filename = os.path.join(ship_dir, f"port_visits_{safe_ship_type}.csv")
        group_df.to_csv(filename, index=False)
        ship_type_csvs[ship_type] = filename

    # Save combined dataset
    combined_filename = os.path.join(
        base_dir, "port_visits_with_sequences_collapsed.csv"
    )
    final_df.to_csv(combined_filename, index=False)
    ship_type_csvs["ALL"] = combined_filename

    print("\n--- CSVs generated by ship type ---")
    for k, v in ship_type_csvs.items():
        print(f"{k}: {v}")
