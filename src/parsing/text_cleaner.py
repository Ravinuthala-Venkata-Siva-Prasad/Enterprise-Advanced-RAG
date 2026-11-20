import re 
import unicodedata 

class TextCleaner:

    @staticmethod
    def clean(text: str)-> str:
        if not text:
            return ""

        text = unicodedata.normalize("NFKC",text)
        text = text.replace("\x00","")
        text = text.replace("\r\n", "\n")
        text = text.replace("\r","\n")
        text = re.sub(r"[ \t]+"," ",text)
        text = re.sub(r"\n{3,}","\n\n",text)

        return text.strip()

    