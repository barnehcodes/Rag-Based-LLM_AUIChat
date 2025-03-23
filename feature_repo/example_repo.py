from feast import Entity
from feast import FeatureView, Field
from datetime import timedelta
from feast import FileSource
from feast.types import Float32, Int64  # for dtypes in schema

driver = Entity(name="driver", join_keys=["driver_id"])


driver_stats_source = FileSource(
    name="driver_hourly_stats_source",
    path="data/driver_stats.parquet",       # path to the file with feature data
    timestamp_field="event_timestamp",      # timestamp column for point-in-time joins
    created_timestamp_column="created"      # when data was added (if available)
)



driver_stats_fv = FeatureView(
    name="driver_hourly_stats",
    entities=[driver],                       # link to the driver entity
    ttl=timedelta(days=1),                   # time-to-live for online cache (1 day)
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
        Field(name="avg_daily_trips", dtype=Int64),
    ],
    online=True,                             # allow materialization to online store
    source=driver_stats_source,              # the batch source defined above
    tags={"team": "driver_performance"}      # optional metadata tags
)
