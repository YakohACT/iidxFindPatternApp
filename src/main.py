import sys
import asyncio
# Install playwright and its browser binaries
#!{sys.executable} -m pip install playwright
#!{sys.executable} -m playwright install
#Install missing system dependencies for the browsers
#!{sys.executable} -m playwright install-deps

from playwright.async_api import async_playwright

async def get_score_urls(index_page_url):
    """一覧ページから譜面のURLを自動抽出する関数(Async版)"""
    extracted_urls = []

    async with async_playwright() as p:
        # Chromiumを起動
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print(f"一覧ページにアクセスしています: {index_page_url}")
        await page.goto(index_page_url)

        # JavaScriptの描画待ち
        await page.wait_for_timeout(3000)

        # ページ内のすべてのリンクを取得
        all_links = await page.evaluate('''() => {
            const links = document.querySelectorAll('a');
            return Array.from(links).map(a => a.href);
        }''')

        # 抽出条件に合うURLを絞り込み
        for link in all_links:
            if "textage.cc/score/" in link and ".html" in link:
                extracted_urls.append(link)

        await browser.close()

    unique_urls = list(set(extracted_urls))
    return unique_urls

# 実行部分

if __name__ == "__main__":
    TARGET_INDEX_URL = "https://textage.cc/score/?a011B000"

    # ノートブック環境では await を直接使用して実行します
    urls = get_score_urls(TARGET_INDEX_URL)

    print(f"\n{len(urls)} 件の譜面URLを自動取得しました！\n")

    for u in urls[:5]:
        print(u)

    print("\n残りのURLを使用して処理を継続できます。")