# Data Handling

Do not commit patient data, public dataset images, DICOM files, masks, or derived images to this repository unless they are explicitly approved and fully anonymized.

Recommended local layout outside version control:

```text
data_local/
  cxr_pneumothorax/
    raw/
    processed/
    masks/
  head_ct_hemorrhage/
    raw_dicom/
    anonymized/
    processed/
    masks/
```

Rules:
- Keep original patient DICOMs outside the repository.
- Anonymize local data before any analysis output is used in the thesis.
- Do not include patient names, IDs, accession numbers, dates of birth, or institution identifiers in figures.
- Use aggregate counts in the thesis whenever possible.
- Store only scripts, protocols, and non-identifiable result summaries in this repository.

