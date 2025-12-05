"""
Google Custom Search API 테스트 스크립트

이 스크립트는 Google Custom Search API를 테스트합니다.
텍스트 검색과 이미지 검색을 모두 지원합니다.
"""

import os
import json
import requests
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# API 설정
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")


def google_custom_search(query, search_type="text", num_results=5):
    """
    Google Custom Search API를 사용하여 검색 수행

    Args:
        query: 검색 키워드
        search_type: 'text' 또는 'image'
        num_results: 결과 개수 (최대 10)

    Returns:
        검색 결과 리스트
    """
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": GOOGLE_SEARCH_ENGINE_ID,
        "q": query,
        "num": num_results,
    }

    if search_type == "image":
        params["searchType"] = "image"

    print(f"\n{'='*60}")
    print(f"🔍 검색 요청: '{query}' (타입: {search_type})")
    print(f"{'='*60}")

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        # API 사용량 정보 출력
        if "searchInformation" in data:
            search_info = data["searchInformation"]
            print(f"\n📊 검색 정보:")
            print(f"   - 검색 시간: {search_info.get('searchTime', 'N/A')}초")
            total_results = search_info.get('totalResults', 'N/A')
            if total_results != 'N/A':
                print(f"   - 총 결과 수: {int(total_results):,}개")
            else:
                print(f"   - 총 결과 수: {total_results}개")

        results = []

        if "items" in data:
            for idx, item in enumerate(data["items"], 1):
                if search_type == "image":
                    result = {
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                        "context": item.get("snippet", ""),
                        "width": item.get("image", {}).get("width", ""),
                        "height": item.get("image", {}).get("height", ""),
                    }
                    print(f"\n{idx}. {result['title']}")
                    print(f"   URL: {result['link']}")
                    print(f"   썸네일: {result['thumbnail']}")
                    print(f"   크기: {result['width']}x{result['height']}")
                else:
                    result = {
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "displayLink": item.get("displayLink", ""),
                    }
                    print(f"\n{idx}. {result['title']}")
                    print(f"   URL: {result['link']}")
                    print(f"   도메인: {result['displayLink']}")
                    print(f"   설명: {result['snippet']}")

                results.append(result)
        else:
            print("\n⚠️  검색 결과가 없습니다.")

        return results

    except requests.exceptions.RequestException as e:
        print(f"\n❌ API 요청 오류: {e}")
        if hasattr(e.response, 'text'):
            print(f"   응답: {e.response.text}")
        return []
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return []


def test_quota():
    """API 할당량 확인"""
    print("\n" + "="*60)
    print("📌 Google Custom Search API 정보")
    print("="*60)
    print(f"API Key: {GOOGLE_SEARCH_API_KEY[:20]}... (길이: {len(GOOGLE_SEARCH_API_KEY) if GOOGLE_SEARCH_API_KEY else 0})")
    print(f"Search Engine ID: {GOOGLE_SEARCH_ENGINE_ID}")
    print("\n💡 참고사항:")
    print("   - 무료 할당량: 하루 100회 검색")
    print("   - 초과 시: 검색당 $5 (1,000회당)")
    print("   - 한 번의 요청당 최대 10개 결과")


def main():
    """메인 테스트 함수"""

    # API 키 확인
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
        print("❌ .env 파일에 API 키가 설정되지 않았습니다!")
        print("\n필요한 환경 변수:")
        print("  - GOOGLE_CUSTOM_SEARCH_API_KEY")
        print("  - GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
        return

    test_quota()

    # 테스트 1: 텍스트 검색
    print("\n\n" + "="*60)
    print("테스트 1: 텍스트 검색")
    print("="*60)
    text_results = google_custom_search("Python programming", search_type="text", num_results=3)

    # 테스트 2: 이미지 검색
    print("\n\n" + "="*60)
    print("테스트 2: 이미지 검색")
    print("="*60)
    image_results = google_custom_search("Python logo", search_type="image", num_results=3)

    # 결과 요약
    print("\n\n" + "="*60)
    print("📋 테스트 요약")
    print("="*60)
    print(f"✓ 텍스트 검색 결과: {len(text_results)}개")
    print(f"✓ 이미지 검색 결과: {len(image_results)}개")

    # JSON 파일로 저장
    save_results = input("\n결과를 JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower()
    if save_results == 'y':
        results_data = {
            "text_search": {
                "query": "Python programming",
                "results": text_results
            },
            "image_search": {
                "query": "Python logo",
                "results": image_results
            }
        }

        with open("search_results.json", "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print("✓ 결과가 search_results.json에 저장되었습니다.")


if __name__ == "__main__":
    main()
