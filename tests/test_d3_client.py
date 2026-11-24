import base64

from scoring.d3_client import placed_request


def test_images_are_placed_assets_in_front_of_the_state(tmp_path):
    frames = []
    for index in range(2):
        path = tmp_path / f"{index}.png"
        path.write_bytes(b"\x89PNG" + bytes([index]))
        frames.append(path)
    questions = {"q": {"type": "noul", "instructions": "A cat?"}}
    request = placed_request("vllm-sr/d3-mini", "the scene", questions, frames)

    assert set(request) == {"model", "state", "questions", "assets"}
    *keys, state = request["state"]
    assert state == "the scene" and len(keys) == 2
    assert list(request["assets"]) == keys
    for key, path in zip(keys, frames):
        asset = request["assets"][key]
        assert asset["type"] == "image"
        assert base64.b64decode(asset["data_base64"]) == path.read_bytes()
        assert not asset["data_base64"].startswith("data:")


def test_text_only_request_has_no_assets():
    request = placed_request("m", "state", {"q": {"type": "noul", "instructions": "x"}})
    assert request == {"model": "m", "state": "state",
                       "questions": {"q": {"type": "noul", "instructions": "x"}}}
