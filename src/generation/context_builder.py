class ContextBuilder:

    def __init__(
        self,
        max_chunks=5,
        max_context_characters=12000
    ):

        self.max_chunks = max_chunks
        self.max_context_characters = max_context_characters

    def build(self, results):

        selected = []
        current_length = 0

        for result in results[:self.max_chunks]:

            document = result["document"]

            text = document["page_content"]
            metadata = document.get("metadata", {})

            source = metadata.get(
                "source",
                metadata.get("file_name", "Unknown source")
            )

            chunk_id = metadata.get(
                "chunk_id",
                "Unknown chunk"
            )

            page = metadata.get("page_number")

            source_text = (
                f"[SOURCE: {source} | CHUNK: {chunk_id}"
                f"{f' | PAGE: {page}' if page is not None else ''}]"
            )

            block = f"{source_text}\n{text}"

            remaining = self.max_context_characters - current_length

            if remaining <= 0:
                break

            if len(block) > remaining:
                if not selected:
                    block = block[:remaining]
                    selected.append(block)
                break

            selected.append(block)
            current_length += len(block) + 2

        return "\n\n".join(selected)