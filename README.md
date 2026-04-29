# Word File Merger

This project includes a simple Python script to merge two Microsoft Word `.docx` files into one output file.

## File

- `merge_word_files.py` merges two `.docx` files in order.

## Requirement

Install the dependencies with:

```powershell
pip install -r requirements.txt
```

## Usage Command

Run the script with:

```powershell
python merge_word_files.py <first_file.docx> <second_file.docx> <output_file.docx>
```

## Example Command

```powershell
python merge_word_files.py "Chapter 1.docx" "second.docx" "merged.docx"
```

## What It Does

- Takes the content from the first Word file
- Appends the content from the second Word file
- Saves the result as a new `.docx` file

## Notes

- This script supports only `.docx` files.
- If the output folder does not exist, it will be created automatically.
