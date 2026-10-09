import argparse

from lib.multimodal_search import image_search_command, verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding generation"
    )
    verify_parser.add_argument("image_path", type=str, help="Path to an image")
    image_parser = subparsers.add_parser(
        "image_search", help="Search movies using an image"
    )
    image_parser.add_argument("image_path", type=str, help="Path to an image")

    args = parser.parse_args()
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            for i, result in enumerate(image_search_command(args.image_path), 1):
                print(f"{i}. {result['title']} (similarity: {result['score']:.4f})")
                print(f"  {result['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
