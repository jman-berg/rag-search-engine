#!/usr/bin/env python3

import argparse
import json


def main() -> None:
    with open("data/movies.json", "r") as f:
        movie_dict = json.load(f)

    results = []

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    for movie in movie_dict["movies"]: 
            if args.query in movie['title']:
                results.append(movie)

    match args.command:
        case "search":
            index = 0
            print(f"Searching for: {args.query}") 
            for result in results[:5]:
                index += 1
                print(f"{index}. Movie {result['title']}")
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
