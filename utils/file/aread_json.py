"""Async file read helper for graph Utils."""


async def aread_content(path, mode="r", j=True):
    import aiofiles

    async with aiofiles.open(path, mode=mode, encoding="utf-8") as f:
        raw = await f.read()
    if j:
        import json
        return json.loads(raw)
    return raw
