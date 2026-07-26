# # # # from pathlib import Path
# # # # import pandas as pd

# # # # ROOT = Path(r"C:\Users\anupa\Downloads\CUAD_v1\CUAD_v1\full_contract_pdf")

# # # # SOURCE_DIRS = [
# # # #     ROOT / "Part_I",
# # # #     ROOT / "Part_II",
# # # #     ROOT / "Part_III",
# # # # ]

# # # # MERGED_DIR = ROOT / "merged"

# # # # missing = []

# # # # total_source = 0

# # # # for src in SOURCE_DIRS:
# # # #     for pdf in src.rglob("*.pdf"):
# # # #         total_source += 1

# # # #         # Expected location in merged
# # # #         dest = MERGED_DIR / pdf.parent.name / pdf.name

# # # #         if not dest.exists():
# # # #             missing.append({
# # # #                 "filename": pdf.name,
# # # #                 "category": pdf.parent.name,
# # # #                 "expected_path": str(dest),
# # # #                 "source_path": str(pdf)
# # # #             })

# # # # print("=" * 60)
# # # # print(f"Total source PDFs : {total_source}")
# # # # print(f"Found in merged   : {total_source - len(missing)}")
# # # # print(f"Missing           : {len(missing)}")
# # # # print("=" * 60)

# # # # if missing:
# # # #     missing_df = pd.DataFrame(missing)
# # # #     missing_df.to_csv(ROOT / "missing_after_copy.csv", index=False)
# # # #     print(missing_df.head(20))
# # # # else:
# # # #     print("✅ All 510 PDFs are present in the merged folder.")
# # # # # from pathlib import Path

# # # # # root = Path(r"C:\Users\anupa\Downloads\CUAD_v1\CUAD_v1\full_contract_pdf\merged")

# # # # # count = sum(
# # # # #     1
# # # # #     for f in root.rglob("*")
# # # # #     if f.is_file() and f.suffix.lower() == ".PDF"
# # # # # )

# # # # # print(count)
# # # from pathlib import Path
# # # import pandas as pd
# # # import json

# # # OUTPUT_DIR = Path(r"C:\Users\anupa\Downloads\CUAD_v1\CUAD_v1\full_contract_pdf\merged")   # Change to your merged folder

# # # rows = []

# # # for folder in OUTPUT_DIR.iterdir():
# # #     if folder.is_dir():
# # #         for pdf in folder.iterdir():
# # #             if pdf.is_file() and pdf.suffix.lower() == ".pdf":
# # #                 rows.append({
# # #                     "Filename": pdf.name,
# # #                     "folder_name": folder.name
# # #                 })
# # # # CSV
# # # df = pd.DataFrame(rows)
# # # df.to_csv("pdf_folder_mapping.csv", index=False)

# # # # JSON
# # # with open("pdf_folder_mapping.json", "w", encoding="utf-8") as f:
# # #     json.dump(rows, f, indent=4)

# # # print(f"Generated mapping for {len(rows)} PDFs.")

# # # import pandas as pd

# # # mapping = pd.read_csv("pdf_folder_mapping.csv")
# # # data = pd.read_csv(r"C:\Users\anupa\Downloads\CUAD_v1\CUAD_v1\master_clauses_updated.csv")


# # # # Normalize filenames (case-insensitive)
# # # data["Filename_key"] = data["Filename"].str.strip().str.lower()
# # # mapping["Filename_key"] = mapping["Filename"].str.strip().str.lower()

# # # # Merge
# # # result = data.merge(
# # #     mapping[["Filename_key", "folder_name"]],
# # #     on="Filename_key",
# # #     how="left"
# # # )
# # # # Manual fixes for known filename mismatches
# # # manual_mapping = {
# # #     "HarpoonTherapeuticsInc_20200312_10-K_EX-10.18_12051356_EX-10.18_Development Agreement.PDF":
# # #         "Development",

# # #     "KALLOINC_11_03_2011-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.PDF'":
# # #         "Strategic Alliance",

# # #     "LECLANCHÉ S.A. - JOINT DEVELOPMENT AND MARKETING AGREEMENT.PDF":
# # #         "Marketing",

# # #     "MACY'S,INC_05_11_2020-EX-99.4-JOINT FILING AGREEMENT.PDF":
# # #         "Joint Venture",

# # #     "MOELIS&CO_03_24_2014-EX-10.19-STRATEGIC ALLIANCE AGREEMENT.PDF":
# # #         "Strategic Alliance",

# # #     "Monsanto Company - SECOND A&R EXCLUSIVE AGENCY AND MARKETING AGREEMENT .PDF":
# # #         "Marketing",

# # #     "PACIRA PHARMACEUTICALS, INC. - A&R STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT .PDF":
# # #         "Marketing",

# # #     "PLAYAHOTELS&RESORTSNV_03_14_2017-EX-10.22-STRATEGIC ALLIANCE AGREEMENT (Hyatt Ziva Cancun).PDF":
# # #         "Strategic Alliance",

# # #     "Reinsurance Group of America, Incorporated - A&R REMARKETING  AGREEMENT.PDF":
# # #         "Marketing",

# # #     "SightLife Surgical, Inc. - STRATEGIC SALES & MARKETING AGREEMENT.PDF":
# # #         "Marketing",
# # # }

# # # # Fill missing folder_name using manual mapping
# # # mask = result["folder_name"].isna()
# # # result.loc[mask, "folder_name"] = (
# # #     result.loc[mask, "Filename"].map(manual_mapping)
# # # )

# # # # Remove temporary key
# # # result.drop(columns=["Filename_key"], inplace=True)

# # # # Save
# # # result.to_csv(r"C:\Users\anupa\Downloads\CUAD_v1\CUAD_v1\master_clauses_updated1.csv", index=False)

# # # # print("Done!")
# # # # import pandas as pd

# # # # mapping = pd.read_csv("pdf_folder_mapping.csv")
# # # # data = pd.read_csv(
# # # #     r"C:\Users\anupa\Downloads\CUAD_v1\CUAD_v1\master_clauses_updated.csv"
# # # # )

# # # # # Normalize filenames (case-insensitive)
# # # # data["Filename_key"] = data["Filename"].astype(str).str.strip().str.lower()
# # # # mapping["Filename_key"] = mapping["Filename"].astype(str).str.strip().str.lower()

# # # # # -------------------------
# # # # # CSV rows with no matching PDF
# # # # # -------------------------
# # # # csv_not_found = data.loc[
# # # #     ~data["Filename_key"].isin(mapping["Filename_key"])
# # # # ]

# # # # print("=" * 60)
# # # # print(f"CSV files without matching PDF: {len(csv_not_found)}")
# # # # print("=" * 60)

# # # # if len(csv_not_found):
# # # #     print(csv_not_found[["Filename"]].drop_duplicates())
# # # #     csv_not_found.to_csv("csv_without_pdf.csv", index=False)

# # # # # -------------------------
# # # # # PDFs with no matching CSV
# # # # # -------------------------
# # # # pdf_not_found = mapping.loc[
# # # #     ~mapping["Filename_key"].isin(data["Filename_key"])
# # # # ]

# # # # print("=" * 60)
# # # # print(f"PDFs without matching CSV: {len(pdf_not_found)}")
# # # # print("=" * 60)

# # # # if len(pdf_not_found):
# # # #     print(pdf_not_found[["Filename", "folder_name"]])
# # # #     pdf_not_found.to_csv("pdf_without_csv.csv", index=False)

# # # # import pandas as pd

# # # # og = pd.read_csv(r"C:\Users\anupa\Downloads\CUAD_v1\CUAD_v1\master_clauses_updated.csv")
# # # # updated = pd.read_csv(r"C:\Users\anupa\Downloads\CUAD_v1\CUAD_v1\master_clauses_updated1.csv")

# # # # # Rows only in updated
# # # # extra = updated.loc[
# # # #     ~updated["Filename"].isin(og["Filename"])
# # # # ]

# # # # print("New unique filenames:", len(extra))
# # # # print(extra)

# # # # # Duplicates in updated
# # # # dups = updated[updated["Filename"].duplicated(keep=False)]

# # # # print("Duplicate rows:", len(dups))
# # # # print(dups.sort_values("Filename"))
# # from pathlib import Path
# # import pandas as pd
# # import os

# # TXT_FOLDER = Path(r"C:\Users\anupa\Downloads\CUAD_v1\CUAD_v1\full_contract_txt")

# # data = pd.read_csv(
# #     r"C:\Users\anupa\Downloads\CUAD_v1\CUAD_v1\master_clauses_updated1.csv"
# # )

# # # -------------------------
# # # Build txt lookup
# # # -------------------------
# # txt_map = {}

# # for txt in TXT_FOLDER.glob("*.txt"):
# #     key = os.path.splitext(txt.name)[0].strip().lower()
# #     txt_map[key] = txt.name

# # # -------------------------
# # # Manual filename fixes
# # # -------------------------
# # manual_mapping = {
# #     "harpoontherapeuticsinc_20200312_10-k_ex-10.18_12051356_ex-10.18_development agreement":
# #         "harpoontherapeuticsinc_20200312_10-k_ex-10.18_12051356_ex-10.18_development agreement_option agreement",

# #     "macy's,inc_05_11_2020-ex-99.4-joint filing agreement":
# #         "macy_s,inc_05_11_2020-ex-99.4-joint filing agreement",

# #     "moelis&co_03_24_2014-ex-10.19-strategic alliance agreement":
# #         "moelis_co_03_24_2014-ex-10.19-strategic alliance agreement",

# #     "playahotels&resortsnv_03_14_2017-ex-10.22-strategic alliance agreement (hyatt ziva cancun)":
# #         "playahotels_resortsnv_03_14_2017-ex-10.22-strategic alliance agreement (hyatt ziva cancun)",

# #     "monsanto company - second a&r exclusive agency and marketing agreement ":
# #         "monsanto company - second a_r exclusive agency and marketing agreement ",

# #     "pacira pharmaceuticals, inc. - a&r strategic licensing, distribution and marketing agreement ":
# #         "pacira pharmaceuticals, inc. - a_r strategic licensing, distribution and marketing agreement ",

# #     "reinsurance group of america, incorporated - a&r remarketing  agreement":
# #         "reinsurance group of america, incorporated - a_r remarketing  agreement",

# #     "sightlife surgical, inc. - strategic sales & marketing agreement":
# #         "sightlife surgical, inc. - strategic sales _ marketing agreement",

# #     "leclanché s.a. - joint development and marketing agreement":
# #         "leclanché s.a. - joint development and marketing agreement",

# #     "kalloinc_11_03_2011-ex-10.1-strategic alliance agreement'":
# #         "kalloinc_11_03_2011-ex-10.1-strategic alliance agreement",
# # }

# # # -------------------------
# # # Match
# # # -------------------------
# # txt_names = []

# # for filename in data["Filename"]:

# #     key = os.path.splitext(str(filename))[0].strip().lower()

# #     if key in manual_mapping:
# #         key = manual_mapping[key]

# #     txt_names.append(txt_map.get(key))

# # data["txt_filename"] = txt_names

# # data.to_csv(
# #     r"C:\Users\anupa\Downloads\CUAD_v1\CUAD_v1\master_clauses_updated2.csv",
# #     index=False
# # )

# # print("Matched:", data["txt_filename"].notna().sum())
# # print("Unmatched:", data["txt_filename"].isna().sum())

# import os 
# import pandas as pd

# METADATA_CSV_PATH = r"C:\Users\anupa\OneDrive\Desktop\Anupam\workshop\Advance_Rag_youtube\CUDA_Rag\DATA\master_clauses_updated_final.csv"

# if os.path.exists(METADATA_CSV_PATH):
#     df_meta = pd.read_csv(METADATA_CSV_PATH, encoding="Windows-1252").fillna("")

#     # Find duplicates
#     duplicate_counts = (
#         df_meta["Filename"]
#         .value_counts()
#         .loc[lambda x: x > 1]
#     )

#     if not duplicate_counts.empty:
#         print(f"\nFound {len(duplicate_counts)} duplicate filenames:\n")

#         for filename, count in duplicate_counts.items():
#             print(f"{filename}  --> {count} times")

#         # Print all duplicate rows (optional)
#         print("\nDuplicate rows:")
#         print(
#             df_meta[df_meta["Filename"].isin(duplicate_counts.index)]
#             .sort_values("Filename")
#         )

#     # Keep first occurrence
#     df_meta = df_meta.drop_duplicates(subset=["Filename"], keep="first")

#     # Create metadata map
#     METADATA_MAP = df_meta.set_index("Filename").to_dict(orient="index")

# else:
#     print("metadata.csv not found.")
#     METADATA_MAP = {}

import logfire
import time

# 1. Configure Logfire (This is the line your previous script was missing)
# 'if-token-present' means it will send to the dashboard if you ran `logfire auth`, 
# otherwise it just runs quietly locally.
logfire.configure(send_to_logfire=False)

# 2. A simple log message
logfire.info("✅ Logfire is successfully configured and running!")

# 3. Testing a span (Useful for timing functions and tracking errors)
def process_mock_file(filename: str):
    # This creates a block in your logs showing exactly how long this took
    with logfire.span("Processing File", file=filename):
        logfire.info(f"Extracting text from {filename}...")
        time.sleep(1) # Simulating work
        
        try:
            if filename == "bad_file.txt":
                raise ValueError("The file is corrupted!")
            logfire.info("Successfully chunked and embedded.")
        except Exception as e:
            logfire.error(f"Failed to process: {e}")

# Run the test
if __name__ == "__main__":
    process_mock_file("good_file.pdf")
    process_mock_file("bad_file.txt")