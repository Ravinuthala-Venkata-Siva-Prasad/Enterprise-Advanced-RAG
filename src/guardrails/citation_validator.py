import re


class CitationValidator:

    CITATION_PATTERN = r"\[SOURCE:\s*([^\]]+)\]"

    def validate(self, answer, retrieved_documents):

        available_sources = set()

        for result in retrieved_documents:

            document = result.get("document", result)

            metadata = document.get("metadata", {})

            source = (
                metadata.get("source")
                or metadata.get("file_name")
            )

            if source:
                available_sources.add(source)

        cited_sources = re.findall(
            self.CITATION_PATTERN,
            answer,
            flags=re.IGNORECASE
        )

        valid_citations = []
        invalid_citations = []

        for citation in cited_sources:

            source_name = citation.split("|")[0].strip()

            if source_name in available_sources:
                valid_citations.append(source_name)
            else:
                invalid_citations.append(source_name)

        return {
            "valid": len(invalid_citations) == 0,
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "citation_count": len(cited_sources)
        }