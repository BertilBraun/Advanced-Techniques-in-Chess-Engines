from __future__ import annotations

import json

from pydantic import BaseModel


def canonical_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode='json'),
        ensure_ascii=False,
        separators=(',', ':'),
        sort_keys=True,
    )
