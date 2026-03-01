#!/usr/bin/env python3

import argparse

from lib.semantic_search import verify_model,embed_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")


    embed_text_parser = subparsers.add_parser("embed_text", help="embeds the piece of text" )
    embed_text_parser.add_argument("text", type=str, help="the text to embed")

    verify_parser = subparsers.add_parser("verify", help="Verify the loaded embeddings model")

    args = parser.parse_args()

    match args.command:
        case "verify":
           verify_model() 
        case "embed_text":
            print("embedding text")
            embed_text(args.text)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
