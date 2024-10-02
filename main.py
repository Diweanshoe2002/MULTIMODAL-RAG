from typing import Optional
from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()

def nse_headers_session(url):
    baseurl = "https://www.nseindia.com/"
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                             'like Gecko) '
                             'Chrome/80.0.3987.149 Safari/537.36',
               'accept-language': 'en,gu;q=0.9,hi;q=0.8', 'accept-encoding': 'gzip, deflate, br'}
    session = requests.Session()
    request = session.get(baseurl, headers=headers)
    cookies = dict(request.cookies)
    response = session.get(url, headers=headers, cookies=cookies)
    raw = response.json()
    return raw

@app.get("/nse-corp-info/")
async def get_nse_corp_info(symbol: str = "OIL", corp_type: str = "announcement", market: str = "equities"):
    url = f"https://www.nseindia.com/api/corp-info?symbol={symbol}&corpType={corp_type}&market={market}"
    try:
        data = nse_headers_session(url)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
