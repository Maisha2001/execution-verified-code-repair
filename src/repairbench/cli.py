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
    models = sub.add_parser("download-models")
    models.add_argument("--models",type=Path,default=Path("data/models"))
    for name in ("doctor","evaluate","compare"):
        child = sub.add_parser(name)
        child.add_argument("--image-file",type=Path,default=Path("runs/image.id"))
        child.add_argument("--wall-seconds",type=float,default=8)
        if name in {"evaluate", "compare"}:
            child.add_argument("--data",type=Path,default=Path("data/quixbugs"))
            child.add_argument("--output",type=Path,required=True)
            child.add_argument("--budget",type=int,default=4 if name == "compare" else 12)
            child.add_argument("--tasks",nargs="+")
        if name == "compare":
            child.add_argument("--models",type=Path,default=Path("data/models"))
            child.add_argument("--device",choices=("cuda","cpu"),default="cuda")
            child.add_argument("--max-new-tokens",type=int,default=768)
    args = parser.parse_args()
    if args.command == "data":
        value = prepare(args.output,args.archive)
        print(json.dumps({"revision":value["revision"],"tasks":len(value["tasks"])}))
    elif args.command == "check-data":
        print(json.dumps({"verified_tasks":len(verify(args.data)["tasks"])}))
    elif args.command == "download-models":
        from .models import download_models
        download_models(args.models)
    else:
        runner = DockerRunner(args.image_file.read_text().strip(),Limits(wall_seconds=args.wall_seconds))
        if args.command == "doctor": print(json.dumps(runner.preflight(),indent=2))
        elif args.command == "evaluate": print(json.dumps(evaluate(args.data,args.output,runner,args.budget,args.tasks),indent=2))
        else:
            from .comparison import compare
            from .inference import CodeModel
            from .retrieval import ContextRetriever
            runner.preflight()
            retriever = ContextRetriever(args.models)
            generator = CodeModel(args.models, args.device, args.max_new_tokens)
            print(json.dumps(compare(args.data,args.output,runner,generator,retriever,args.budget,args.tasks),indent=2))


if __name__ == "__main__":
    main()
