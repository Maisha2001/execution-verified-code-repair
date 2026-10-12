import argparse
import json
from pathlib import Path

from .data import prepare, verify
from .evaluate import evaluate
from .sandbox import DockerRunner, Limits


def main():
    parser = argparse.ArgumentParser(description="Isolated Python repair benchmark")
    sub = parser.add_subparsers(dest="command",required=True)
    data = sub.add_parser("data")
    data.add_argument("--output",type=Path,default=Path("data/quixbugs"))
    data.add_argument("--archive",type=Path)
    check = sub.add_parser("check-data")
    check.add_argument("--data",type=Path,default=Path("data/quixbugs"))
    for name in ("doctor","evaluate"):
        child = sub.add_parser(name)
        child.add_argument("--image-file",type=Path,default=Path("runs/image.id"))
        child.add_argument("--wall-seconds",type=float,default=8)
        if name == "evaluate":
            child.add_argument("--data",type=Path,default=Path("data/quixbugs"))
            child.add_argument("--output",type=Path,required=True)
            child.add_argument("--budget",type=int,default=12)
            child.add_argument("--tasks",nargs="+")
    args = parser.parse_args()
    if args.command == "data":
        value = prepare(args.output,args.archive)
        print(json.dumps({"revision":value["revision"],"tasks":len(value["tasks"])}))
    elif args.command == "check-data":
        print(json.dumps({"verified_tasks":len(verify(args.data)["tasks"])}))
    else:
        runner = DockerRunner(args.image_file.read_text().strip(),Limits(wall_seconds=args.wall_seconds))
        if args.command == "doctor": print(json.dumps(runner.preflight(),indent=2))
        else: print(json.dumps(evaluate(args.data,args.output,runner,args.budget,args.tasks),indent=2))


if __name__ == "__main__":
    main()
