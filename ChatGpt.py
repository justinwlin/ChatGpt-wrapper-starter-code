import openai
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter


class OpenAIChatGPT:
    def __init__(self, api_key, default_model="gpt-3.5-turbo-1106", max_tokens=16385):
        self.client = openai.OpenAI(api_key=api_key)
        self.default_model = default_model
        self.max_tokens = max_tokens
        self.tokenizer = Tokenizer(default_model)
        self.api_key = api_key

    def split_text(self, text):
        """
        Splits the text into chunks based on the token limit.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_tokens,
            length_function=self.tokenizer.count_token,
            separators=["\n\n", "\n", " ", ""],
            chunk_overlap=2,
        )
        return splitter.split_text(text)

    def create_chat_message(self, model, system_prompt, user_prompt):
        if model is None:
            model = self.default_model

        # Handle long prompts by splitting
        if self.tokenizer.count_token(user_prompt) > self.max_tokens:
            responses = []
            for chunk in self.split_text(user_prompt):
                response = self._send_chat_request(model, system_prompt, chunk)
                responses.append(response)
            return " ".join(responses)
        else:
            return self._send_chat_request(model, system_prompt, user_prompt)

    def _send_chat_request(self, model, system_prompt, prompt_chunk):
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_chunk},
            ],
        )

        response_text = response.choices[0].message.content
        self.tokenizer.sent_tokens += self.tokenizer.count_token(prompt_chunk)
        self.tokenizer.received_tokens += self.tokenizer.count_token(response_text)
        return response_text

    def report_token_usage(self):
        self.tokenizer.report_token_usage()


class Tokenizer:
    def __init__(self, model_name):
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.sent_tokens = 0
        self.received_tokens = 0

    def count_token(self, text):
        return len(self.encoding.encode(text))

    def report_token_usage(self):
        print(f"Tokens sent: {self.sent_tokens}")
        print(f"Tokens received: {self.received_tokens}")


# Example usage
# if __name__ == "__main__":
#     API_KEY = "YOUR_API_KEY"
#     chat = OpenAIChatGPT(api_key=API_KEY)

#     # Example long user prompt
#     long_user_prompt = (
#         "..."
#     )

#     result = chat.create_chat_message(
#         model=None,
#         system_prompt="You are an assistant...",
#         user_prompt=long_user_prompt,
#     )
#     print(result)
#     chat.report_token_usage()
