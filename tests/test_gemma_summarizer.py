from types import SimpleNamespace

import summarys.gemma_summarizer as gemma_summarizer


class _FakeOpenAI:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self._create,
            )
        )
        self.request_args = None

    def _create(self, **kwargs):
        self.request_args = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="A concise generated summary."),
                )
            ]
        )


def test_summarize_frames_with_gemma_returns_body_with_metadata(monkeypatch):
    fake_client = _FakeOpenAI(base_url="http://unused", api_key="unused")
    monkeypatch.setattr(gemma_summarizer, "OpenAI", lambda **kwargs: fake_client)
    monkeypatch.setenv("SUMMARY_MAX_TOKENS", "123")

    result = gemma_summarizer.summarize_frames_with_gemma(
        frame_descriptions=[{"description": "A red car waits at an intersection."}],
        timestamps=[2.5],
        style="concise",
        model="gemma-4-26B-MoE",
        base_url="http://10.20.1.28:8010/v1",
        api_key="token",
        vision_model="Cosmos-Reason2-8B",
    )

    assert "summary_template:cosmos_summary_v1" in result
    assert "engine:gemma4" in result
    assert "A concise generated summary." in result
    assert fake_client.request_args is not None
    assert fake_client.request_args["model"] == "gemma-4-26B-MoE"
    assert fake_client.request_args["max_tokens"] == 123
    assert "[2.50s] A red car waits at an intersection." in fake_client.request_args["messages"][0]["content"]


def test_summarize_frames_with_gemma_handles_empty_transcript_without_api_call(monkeypatch):
    class _ClientNoChatAllowed:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: (_ for _ in ()).throw(
                        AssertionError("Chat completion should not be called for empty transcript")
                    )
                )
            )

    monkeypatch.setattr(gemma_summarizer, "OpenAI", _ClientNoChatAllowed)

    result = gemma_summarizer.summarize_frames_with_gemma(
        frame_descriptions=[],
        timestamps=[],
    )

    assert "_No content to summarize._" in result
    assert "engine:gemma4" in result
