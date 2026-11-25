from pydantic import BaseModel, ConfigDict


class EmptyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
