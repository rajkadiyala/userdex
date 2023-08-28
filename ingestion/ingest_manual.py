"""Module providing filesystem output."""
import os
import sys
import fitz

def get_file_name(filepath):
    """Gets a filename from the path"""
    return os.path.basename(filepath)

def ingest_py_mu_pdf(filepath):
    """ Converts a filepath to the file's content"""
    text = ""
    doc = fitz.open(filepath)
    for page in doc:
        text+=page.get_text()
    return text

def output_manual_txt_content(text, output_filepath):
    """ Outputs text to a given filepath by overwriting the file"""
    output_file = open(output_filepath, "w",encoding="utf-8")
    output_file.write(text)
    output_file.close()

def main(*args):
    """
    Main ingestion function. Takes in a filepath, extracts text, and outputs it to a file
    in the ingestion/output directory
    """
    filepath = args[0][0]
    print(filepath)
    output = ingest_py_mu_pdf("manuals/auto/ford_ecosport_2020.pdf")
    filename = get_file_name(filepath)
    filename_without_extension = os.path.splitext(filename)[0]
    output_filepath = "/workspaces/userdex/ingestion/output/" + filename_without_extension + ".txt"
    output_manual_txt_content(output, output_filepath)

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
