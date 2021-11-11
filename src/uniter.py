
import os


def unite(dir: str, out: str) -> None:
    """
    Unites all files in a directory into a single file.
    """
    with open(out, 'w') as outfile:
        for filename in os.listdir(dir):
            with open(os.path.join(dir, filename)) as infile:
                outfile.write(infile.read())


if __name__ == '__main__':
    unite("../dataset/processed_games", "../dataset/nm_games.csv")
