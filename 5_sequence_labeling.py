import spacy
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import re
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# RE Patterns
import re

patterns = {

    "course_code": re.compile(
        r'\b[A-Z]{2,6}[\s\-]?\d{3,4}[A-Z]?\b'
    ),

    "instructor_name": re.compile(
        r'\b(Dr\.?|Prof\.?|Professor|Mr\.?|Mrs\.?|Ms\.?|Associate Professor|Assistant Professor)'
        r'\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?' 
        r'(?:\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?)?\b'
    ),

    "department": re.compile(
        r'\b(Department|Dept\.?|School|College|Faculty|Division|Institute|Center|Centre)'
        r'\s+of\s+[A-Z][A-Za-z\s&,]+?(?=\s{2,}|\.|,|;|\n|and\s+[A-Z]|$)',
        re.MULTILINE
    ),

    "date": re.compile(
        r'\b('
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
        r'\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?'
        r'|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}'
        r'|\d{1,2}\s+'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'\.?\s*\d{0,4}'
        r'|(?:Fall|Spring|Summer|Winter)\s+(?:Semester\s+|Term\s+)?\d{4}'
        r'|(?:Academic\s+Year\s+)?\d{4}[\/\-]\d{2,4}'
        r')\b',
        re.IGNORECASE
    ),

    "office": re.compile(
        r'\b('
        r'(?:Office\s+of(?:\s+the)?\s+[A-Z][A-Za-z\s]+?(?=\s*[,.\n]|$))'
        r"|(?:[A-Z][A-Za-z]+(?:'s)?\s+Office)"
        r'|(?:(?:Room|Rm\.?|Suite|Bldg\.?|Building)\s+[A-Z]?\d+[A-Z]?)'
        r'|(?:[A-Z][A-Za-z\s]+(?:Hall|Building|Center|Centre|Tower|House|Block|Wing))'
        r')',
        re.MULTILINE
    ),

    "penalties/grading": re.compile(
        r'\b('
        r'\d+(?:\.\d+)?%\s*(?:deduction|penalty|reduction|late\s+penalty|per\s+day|mark|marks|points?)'
        r'|(?:final\s+)?(?:exam|assignment|quiz|project|essay|midterm|homework)\s+'
        r'(?:weight(?:age)?|worth|value|counts?)\s+(?:for\s+)?\d+(?:\.\d+)?%'
        r')',
        re.IGNORECASE
    ),
}
course_code_pattern = re.compile(r"\b[A-Z]{2,4}\s?\d{3,4}")

if __name__ == "__main__":

    # Loads the documents in from my data/raw directory
    loader = DirectoryLoader(
        path="data/raw/",          
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    space_obj = spacy.load("en_core_web_sm")
    texts = [doc.page_content for doc in documents]

    
    # Loop through the corpus with spacy and find the parts of speech + NE
    parts_of_speech = defaultdict(list)
    named_entities = []
    for _, spacy_doc in zip(documents, space_obj.pipe(texts, batch_size=32)):
        # Adds the token to the parts of speech 
        for token in spacy_doc:
            parts_of_speech[token.pos_].append(token.text)

        for entity in spacy_doc.ents:
            for class_, pattern in patterns.items():
                matches = pattern.findall(entity.text)
                for m in matches:
                    named_entities.append((entity.label, class_, entity.text.replace('\n', "")))

    # Show results
    for pos, list_ in parts_of_speech.items():
        print(f"Number of {pos}s: {len(list_)}")

    ne_df = pd.DataFrame(named_entities, columns=["Label", "Class","Text"])
    print(ne_df.head(15))
    
    