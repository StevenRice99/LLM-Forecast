import argparse
import os


def load_trusted() -> list[str]:
    """
    Load our trusted publishers.
    :return: The trusted publishers.
    """
    path = os.path.join(os.getcwd(), "Trusted.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            trusted = [line.strip() for line in f]
    else:
        trusted = []
    return trusted


def get_publishers(trusted: bool = False) -> None:
    """
    Print a list of publishers to help debug
    :param trusted: If we are interested in trusted sites or not.
    :return:
    """
    # If the data folder does not exist, do nothing.
    path = os.path.join(os.getcwd(), "Data")
    if not os.path.exists(path):
        return
    # Load our trusted files.
    trusted_publishers = load_trusted()
    # Get all publishers.
    publishers = []
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for file in files:
        # We only want the lines with publishers.
        with open(os.path.join(path, file), "r") as f:
            lines = [line.replace("Publisher: ", "").strip() for line in f if line.startswith("Publisher: ")]
        # Check every publisher and add them if they meet the criteria.
        for publisher in lines:
            if (publisher not in publishers and
                    ((trusted and publisher in trusted_publishers) or
                     (not trusted and publisher not in trusted_publishers))):
                publishers.append(publisher)
    # Sort and display the publishers.
    publishers.sort()
    for publisher in publishers:
        print(publisher)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publisher List")
    parser.add_argument("-t", "--trusted", action="store_true", help="Get trusted publishers.")
    args = parser.parse_args()
    get_publishers(args.trusted)
