import re
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents.base import Document


def find_instances(documents: list[Document]) -> dict[str, list]:
    instances = {
        "emails": [],
        "urls": [],
        "dates": [],
        "phone_numbers": [],
        "course_codes": [],
        "sections": [],
    }
    # RE Patterns
    email_pattern = re.compile(r"\b[\w-]+@[\w-]+\.(?:com|org|edu|net)")
    url_pattern = re.compile(
        r"\b(?:https://)?(?:[\w-]+)\.(?:com|org|edu|net)(?:[\w/\.-]+)"
    )
    dates_pattern = re.compile(r"(?:\d{2}[/-]\d{2}[-/]\d{4}|\d{4}[/-]\d{2}[-/]\d{2})")
    phone_numbers_pattern = re.compile(
        r"(?:\+\d{1,3}\s)?(?:\(?\d\d\d\)?[\s-])?\d{3}[\s-]\d{4}"
    )
    course_code_pattern = re.compile(r"\b[A-Z]{2,4}\s?\d{3,4}")
    section_pattern = re.compile(r"(?:Section|Part|Chapter)\s?\d+\.\d+", re.IGNORECASE)

    # Loop through documents
    for document in documents:
        text = document.page_content

        # update the output with all instances
        instances["emails"].extend(email_pattern.findall(text))
        instances["urls"].extend(url_pattern.findall(text))
        instances["dates"].extend(dates_pattern.findall(text))
        instances["phone_numbers"].extend(phone_numbers_pattern.findall(text))
        instances["course_codes"].extend(course_code_pattern.findall(text))
        instances["sections"].extend(section_pattern.findall(text))

    return instances


if __name__ == "__main__":
    loader = DirectoryLoader(path="data/raw/", glob="*.pdf", loader_cls=PyPDFLoader)  # type: ignore

    documents = loader.load()

    instances = find_instances(documents=documents)

    # Print pattern counts
    for pattern, instance_ls in instances.items():
        print("Number of ", pattern, ": ", len(instance_ls))

    # Print Examples In each category
    print("\n=============== EXAMPLES ================")
    for pattern, instance_ls in instances.items():
        print(pattern)
        for instance in instance_ls[:5]:
            print("   ", instance)
        print("-----------------")
