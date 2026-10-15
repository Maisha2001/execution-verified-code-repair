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
    training_data = sub.add_parser("training-data")
    training_data.add_argument("--source", type=Path, default=Path("data/mbpp/sanitized-mbpp.json"))
    training_data.add_argument("--benchmark", type=Path, default=Path("data/quixbugs"))
    training_data.add_argument("--output", type=Path, required=True)
    training_data.add_argument("--image-file", type=Path, default=Path("runs/image.id"))
    training = sub.add_parser("train")
    training.add_argument("--models", type=Path, default=Path("data/models"))
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--resume", type=Path)
    training.add_argument("--stop-after-steps", type=int)
    training.add_argument("--max-seconds", type=float, default=7200)
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--max-tokens", type=int, default=1024)
    for name in ("doctor","evaluate","compare","repair"):
        child = sub.add_parser(name)
        child.add_argument("--image-file",type=Path,default=Path("runs/image.id"))
        child.add_argument("--wall-seconds",type=float,default=8)
        if name in {"evaluate", "compare", "repair"}:
            child.add_argument("--data",type=Path,default=Path("data/quixbugs"))
            child.add_argument("--output",type=Path,required=True)
            child.add_argument("--budget",type=int,default=12 if name == "evaluate" else 4)
            child.add_argument("--tasks",nargs="+")
        if name in {"compare", "repair"}:
            child.add_argument("--adapter",type=Path)
            child.add_argument("--resume",action="store_true")
            child.add_argument("--max-seconds",type=float)
            child.add_argument("--models",type=Path,default=Path("data/models"))
            child.add_argument("--device",choices=("cuda","cpu"),default="cuda")
            child.add_argument("--max-new-tokens",type=int,default=768)
        if name == "repair":
            child.add_argument("--task-seconds",type=float,default=600)
    args = parser.parse_args()
    if args.command == "data":
        value = prepare(args.output,args.archive)
        print(json.dumps({"revision":value["revision"],"tasks":len(value["tasks"])}))
    elif args.command == "check-data":
        print(json.dumps({"verified_tasks":len(verify(args.data)["tasks"])}))
    elif args.command == "download-models":
        from .models import download_models
        download_models(args.models)
    elif args.command == "training-data":
        from .training_data import build_training_data
        runner = DockerRunner(args.image_file.read_text().strip())
        print(json.dumps(build_training_data(args.source, args.benchmark, args.output, runner), indent=2))
    elif args.command == "train":
        from .training import TrainingConfig, train
        settings = TrainingConfig(epochs=args.epochs, max_tokens=args.max_tokens)
        print(json.dumps(train(args.models, args.data, args.output, settings, resume=args.resume,
                               stop_after_steps=args.stop_after_steps, max_seconds=args.max_seconds), indent=2))
    else:
        runner = DockerRunner(args.image_file.read_text().strip(),Limits(wall_seconds=args.wall_seconds))
        if args.command == "doctor": print(json.dumps(runner.preflight(),indent=2))
        elif args.command == "evaluate": print(json.dumps(evaluate(args.data,args.output,runner,args.budget,args.tasks),indent=2))
        else:
            from .comparison import compare
            from .inference import CodeModel
            from .retrieval import ContextRetriever
            runner.preflight()
            generator = CodeModel(args.models, args.device, args.max_new_tokens, adapter=args.adapter)
            if args.command == 'repair':
                from .feedback_evaluation import evaluate_feedback
                print(json.dumps(evaluate_feedback(args.data,args.output,runner,generator,args.budget,args.tasks,
                    resume=args.resume,max_seconds=args.max_seconds if args.max_seconds is not None else 3600,
                    task_seconds=args.task_seconds),indent=2))
            else:
                retriever = ContextRetriever(args.models)
                print(json.dumps(compare(args.data,args.output,runner,generator,retriever,args.budget,args.tasks,
                                         resume=args.resume,max_seconds=args.max_seconds),indent=2))


if __name__ == "__main__":
    main()
