import distyolo


def main() -> None:
    model = distyolo.load_model("yolo11s.pt")

    print(model)

if __name__ == "__main__":
    main()