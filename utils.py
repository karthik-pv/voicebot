# fetch.py

import os
import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}


def create_amazon_url(search_query):
    search_query_array = search_query.split(" ")
    search_url = f"https://www.amazon.in/s?k={'+'.join(search_query_array)}"
    return search_url


def fetch_product_info(search_query):
    url = create_amazon_url(search_query)
    html_content = make_fetch_request(url, headers)
    if html_content:
        relevant_content = parse_html_string_and_fetch_relevant_content(html_content)
        if relevant_content:
            return process_product_string(relevant_content, search_query)
    return None


def make_fetch_request(url, headers):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        return response.text
    except requests.RequestException as error:
        print(f"Error fetching the HTML: {error}")
        return None


def parse_html_string_and_fetch_relevant_content(html_content):
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        search_results = soup.select('[data-component-type="s-search-result"]')

        data_string = ""
        for result in search_results[:10]:  # Limit to 10 results
            data_string += result.get_text(strip=True) + "\n"
            data_string += "-------------------------------------------------------------------------------------------------------\n"

        return data_string
    except Exception as error:
        print(f"Something went wrong: {error}")
        return None


def process_product_string(input_string, title):
    products = input_string.split(
        "-------------------------------------------------------------------------------------------------------"
    )
    product_list = []

    for product in products:
        product = product.replace(
            "SponsoredSponsoredYou are seeing this ad based on the product’s relevance to your search query.Let us know",
            "",
        )
        trimmed_product = product.strip()
        first_rupee_index = trimmed_product.find("₹")
        if first_rupee_index == -1:
            continue

        cost_start_index = first_rupee_index + 1
        cost_end_index = trimmed_product.find("₹", cost_start_index)

        if cost_end_index == -1:
            cost = trimmed_product[cost_start_index:].strip()
        else:
            cost = trimmed_product[cost_start_index:cost_end_index].strip()

        stars_index = trimmed_product.find("stars")
        description = (
            trimmed_product[:first_rupee_index].strip()
            if stars_index == -1
            else trimmed_product[: stars_index + 5].strip()
        )

        product_object = description + " cost -  " + cost + " rupees "
        print(product_object)

        product_list.append(product_object)

    return product_list if product_list else None
