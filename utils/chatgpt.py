from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

import openai
from openai import AsyncOpenAI
from loguru import logger

T = TypeVar("T")


async def _retry_on_error(
    openai_call: Callable[[], Awaitable[T]],
    max_num_trials: int = 5,
    first_wait_time: int = 10,
) -> Awaitable[T] | None:
    """
    API送信時にエラーが発生した場合にリトライするための関数です。
    """
    for i in range(max_num_trials):
        try:
            return await openai_call()
        except openai.APIError as e:  # noqa: PERF203
            if i == max_num_trials - 1:
                raise
            logger.warning(f"We got an error: {e}")
            wait_time_seconds = first_wait_time * (2**i)
            logger.warning(f"Wait for {wait_time_seconds} seconds...")
            await asyncio.sleep(wait_time_seconds)
    return None


class OpenAIChatAPI:
    """
    OpenAI APIのラッパーです。
    `batch_generate_chat_response`メソッドを使用して、複数のチャットリクエストを並列で送信できます。

    Args:
        model: 使用するモデルの名前。
        api_headers: OpenAI APIに送信するリクエストのヘッダー。
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",  # ハッカソン中は gpt-4o-mini 以外使用禁止です
        api_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        if api_headers is None:
            api_headers = {}
        self._client = AsyncOpenAI(**api_headers)

    async def _async_batch_run_chatgpt(
        self,
        messages_list: list[list[dict[str, str]]],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[str]:
        """Send multiple chat requests to the OpenAI in parallel."""
        if stop_sequences is not None:
            if "stop" in kwargs:
                msg = (
                    "You specified both `stop_sequences` and `stop` in generation kwargs. "
                    "However, `stop_sequences` will be normalized into `stop`. "
                    "Please specify only one of them."
                )
                raise ValueError(msg)
            kwargs["stop"] = stop_sequences

        if max_new_tokens is not None:
            if "max_tokens" in kwargs:
                msg = (
                    "You specified both `max_new_tokens` and `max_tokens` in generation kwargs. "
                    "However, `max_new_tokens` will be normalized into `max_tokens`. "
                    "Please specify only one of them."
                )
                raise ValueError(msg)
            kwargs["max_tokens"] = max_new_tokens

        tasks = [
            _retry_on_error(
                # Define an anonymous function with a lambda expression and pass it,
                # and call it inside the _retry_on_error function
                openai_call=lambda x=ms: self._client.chat.completions.create(
                    model=self.model,
                    messages=x,
                    **kwargs,
                ),
            )
            for ms in messages_list
        ]
        return await asyncio.gather(*tasks)

    def batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[str]:
        api_responses = asyncio.run(
            self._async_batch_run_chatgpt(chat_messages_list, **kwargs),
        )
        for res in api_responses:
            logger.info(res.usage)
        return [res.choices[0].message.content for res in api_responses]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
