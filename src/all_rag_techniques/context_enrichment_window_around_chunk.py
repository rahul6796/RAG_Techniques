
import os
import sys

from langchain.docstore.document import Document
from utils import read_pdf_to_string


path = "./data/Understanding_Climate_Change.pdf"


content = read_pdf_to_string(path=path)

