from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
import re
import io

app = FastAPI()

# --- Header parsing with uppercase heuristic and ordinal exception ---
def parse_header(header_str: str):
    tokens = header_str.split()
    period_idx = next((i for i, tok in enumerate(tokens) if re.match(r"\d{2}/\d{2}/\d{4}", tok)), len(tokens))
    prefix = tokens[:period_idx]
    period = ' '.join(tokens[period_idx:])

    idx = 0
    surname_parts = []
    while idx < len(prefix) and prefix[idx].isupper():
        surname_parts.append(prefix[idx])
        idx += 1
    surname = ' '.join(surname_parts)
    first_name = prefix[idx] if idx < len(prefix) else ''
    idx += 1

    ordinal = re.compile(r"^\d+(st|nd|rd|th)$", re.IGNORECASE)
    vessel_start = idx
    for j in range(idx, len(prefix)):
        tok = prefix[j]
        if (tok.isupper() or re.search(r"\d", tok)) and not ordinal.match(tok):
            vessel_start = j
            break

    function = ' '.join(prefix[idx:vessel_start])
    vessel = ' '.join(prefix[vessel_start:])
    return surname, first_name, function, vessel, period

# --- Core processing logic ---
def process_timesheet_df(raw: pd.DataFrame) -> pd.DataFrame:
    df_dates = raw.copy()
    df_dates[0] = pd.to_datetime(df_dates[0], dayfirst=True, errors='coerce')

    header_rows = raw.index[raw.iloc[:,3].astype(str).str.contains('Name:', na=False)].tolist()
    date_rows   = raw.index[raw.iloc[:,0] == 'Date'].tolist()
    totals_rows = raw.index[raw.apply(lambda r: r.astype(str).str.contains('Totals', na=False).any(), axis=1)].tolist()

    sub_heads = [
        "Allowance:", "Deductions:", "Reimbursements:",
        "Overtime Previous Month:", "Traveltime Previous Month:", "General Remarks:"
    ]
    records = []

    for hdr in header_rows:
        mtext = raw.iat[hdr,12] or ""
        surname, first_name, func, vessel, period = parse_header(mtext)

        date_idx    = next(r for r in date_rows if r > hdr)
        codes_start = date_idx + 1
        totals_idx  = next(r for r in totals_rows if r > date_idx)

        row2 = raw.iloc[hdr+3].astype(str).tolist()
        is_type2 = 'Cutter' in row2

        start = codes_start + (1 if is_type2 else 0)
        block = raw.iloc[start:totals_idx]

        codes = block.iloc[:,2].astype(str)
        counts = {c: codes.str.count(f'^{c}$').sum() for c in ['T','W','C','L','S','A']}

        dates = df_dates.iloc[start:totals_idx,0]
        travel = dates[codes=='T'].dt.strftime('%d/%m/%Y').dropna()
        travel_cell = '\n'.join(travel.tolist())

        ot = int(pd.to_numeric(block.iloc[:,6],errors='coerce').fillna(0).sum())
        tr = int(pd.to_numeric(block.iloc[:,7],errors='coerce').fillna(0).sum())

        grp_hdr = raw.iloc[hdr+2].astype(str).tolist()
        rem_col = next((j for j,v in enumerate(grp_hdr) if v=='Remarks'), None)
        rem_vals = [v.strip() for v in block.iloc[:,rem_col] if isinstance(v,str) and v.strip()] if rem_col is not None else []
        rem_cell = '\n'.join(rem_vals)

        cut = block.iloc[:,row2.index('Cutter')].notna().sum() if is_type2 else 0
        sleep = block.iloc[:,row2.index('Sleep')].notna().sum() if is_type2 else 0
        sea = block.iloc[:,row2.index('Sea Passage')].notna().sum() if is_type2 else 0

        allowance_row = next((r for r in range(hdr, raw.shape[0]) if isinstance(raw.iat[r,13], str) and 'Allowance:' in raw.iat[r,13]), None)
        allow_txt = raw.iat[allowance_row,13] if allowance_row is not None else ''
        parsed_allow = {sh[:-1]:'' for sh in sub_heads}
        if allow_txt:
            txt = re.sub(r'\s+', ' ', allow_txt)
            segs = re.split('(?=' + '|'.join(map(re.escape, sub_heads)) + ')', txt)
            for s in segs:
                for sh in sub_heads:
                    if s.startswith(sh): parsed_allow[sh[:-1]] = s[len(sh):].strip(); break

        records.append({
            'Name':f"{surname} {first_name}".strip(),
            'Function':func, 'Vessel':vessel, 'Period':period,
            'Travel Dates':travel_cell, **counts,
            'Overtime':ot, 'Travel':tr, 'Remarks':rem_cell,
            'Cutter':cut, 'Sleep':sleep, 'Sea Passage':sea,
            'Allowance':parsed_allow['Allowance'], 'Deductions':parsed_allow['Deductions'],
            'Reimbursements':parsed_allow['Reimbursements'],
            'Overtime Previous Month':parsed_allow['Overtime Previous Month'],
            'Traveltime Previous Month':parsed_allow['Traveltime Previous Month'],
            'General Remarks':parsed_allow['General Remarks']
        })

    cols = [
        'Name','Function','Vessel','Period','Travel Dates','T','W','C','L','S','A',
        'Overtime','Travel','Remarks','Cutter','Sleep','Sea Passage',
        'Allowance','Deductions','Reimbursements','Overtime Previous Month',
        'Traveltime Previous Month','General Remarks'
    ]
    return pd.DataFrame(records, columns=cols)

# --- Retool-ready endpoints using FastAPI ---
@app.post("/process/json")
async def process_json(file: UploadFile = File(...)):
    try:
        raw = pd.read_excel(file.file, header=None, dtype=str)
        df = process_timesheet_df(raw)
        return JSONResponse(content=df.to_dict(orient='records'))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/process/excel")
async def process_excel(file: UploadFile = File(...)):
    try:
        raw = pd.read_excel(file.file, header=None, dtype=str)
        df = process_timesheet_df(raw)
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        buf.seek(0)
        return StreamingResponse(buf, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers={'Content-Disposition':'attachment; filename=summary.xlsx'})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# To run: uvicorn time_sheet_processing:app --reload
