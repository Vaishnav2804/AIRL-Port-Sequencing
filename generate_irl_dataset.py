import aisdb
from aisdb.database.dbconn import PostgresDBConn
from aisdb.database.dbqry import DBQuery
from aisdb.database import sqlfcn_callbacks
from aisdb.discretize.h3 import Discretizer
from aisdb.ports.api import WorldPortIndexClient

from datetime import datetime, timedelta
import pandas as pd

# -------------------------------
# 1. PostgreSQL Connection Setup
# -------------------------------
db_user = "vaishnav"
db_dbname = "aisviz"
db_password = "<>"
db_hostaddr = "127.0.0.1"
db_port = 5432

start_time = datetime(2022, 1, 1, 0, 0, 0)
end_time = datetime(2022, 1, 3, 23, 59, 59)

xmin, ymin, xmax, ymax = -70, 45, -58, 53

# -------------------------------
# 2. World Port Index (reference data)
# -------------------------------
client = WorldPortIndexClient()
df_ports = client.fetch_ports(lat_min=45.0, lat_max=51.5, lon_min=-71.5, lon_max=-55.0)

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
        callback=sqlfcn_callbacks.in_timerange_validmmsi,
    )

    # Raw generator
    rowgen = qry.gen_qry()

    # Trajectories (decimated for speed)
    tracks = aisdb.track_gen.TrackGen(rowgen, decimate=True)

    # Add H3 index to tracks
    tracks_discretized = discretizer.yield_tracks_discretized_by_indexes(tracks)

    # -------------------------------
    # 4. Stop detection
    # -------------------------------
    SPEED_THRESHOLD_KNOTS = 0.5
    MIN_STOP_DURATION_MIN = 30

    stops = []

    for track in tracks_discretized:
        mmsi = track["mmsi"]
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
                                "start_time": int(first_t),
                                "end_time": int(last_t),
                                "duration_min": dur,
                                "h3_index": current[0][1],
                            }
                        )
                    current = []

        # handle end-of-track stops
        if current:
            first_t, last_t = current[0][0], current[-1][0]
            dur = (int(last_t) - int(first_t)) / 60
            if dur >= MIN_STOP_DURATION_MIN:
                stops.append(
                    {
                        "mmsi": mmsi,
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

    # Drop non-port stops
    stops_df = stops_df.dropna(subset=["PORT_NAME"])

    # Sort chronologically
    stops_df = stops_df.sort_values(["mmsi", "start_time"]).reset_index(drop=True)

    # -------------------------------
    # 6. Build collapsed port & H3 sequences
    # -------------------------------

    def collapse_consecutive(seq_list):
        """Collapse consecutive duplicates in a list."""
        if not seq_list:
            return []
        collapsed = [seq_list[0]]
        for item in seq_list[1:]:
            if item != collapsed[-1]:
                collapsed.append(item)
        return collapsed

    # Build overall collapsed port sequence per MMSI
    def build_port_sequence(group):
        ports = group["PORT_NAME"].tolist()
        collapsed_ports = collapse_consecutive(ports)
        return ",".join(collapsed_ports)

    # Build overall collapsed H3 sequence per MMSI
    def build_h3_sequence(group):
        h3s = group["h3_index"].tolist()
        collapsed_h3s = collapse_consecutive(h3s)
        return ",".join(collapsed_h3s)

    # Compute sequence columns
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

    # Merge back into stop-level table
    stops_with_seq = stops_df.merge(port_seq_df, on="mmsi", how="left")
    stops_with_seq = stops_with_seq.merge(h3_seq_df, on="mmsi", how="left")

    # Convert times to readable datetime
    stops_with_seq["start_time"] = pd.to_datetime(
        stops_with_seq["start_time"], unit="s"
    )
    stops_with_seq["end_time"] = pd.to_datetime(stops_with_seq["end_time"], unit="s")

    # Select final columns
    final_df = stops_with_seq[
        [
            "mmsi",
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

    # Save
    final_df.to_csv("port_visits_with_sequences_collapsed.csv", index=False)
