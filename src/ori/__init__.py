from .app import parse_cli_args, run_app

def main():
    args = parse_cli_args()
    
    run_app(
        target_dir=args.target_dir,
        days_old = args.days_old,
        reset_cache=args.reset,
        allow_siril=args.allow_siril
    )