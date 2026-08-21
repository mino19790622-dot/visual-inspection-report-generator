# run_agent.py
"""Run the LangGraph inspection agent on an image (with decision logic)."""

import argparse
from app.agent.graph import InspectionAgent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LangGraph inspection agent: adaptive detection + risk-based RAG")
    parser.add_argument("image", nargs="?", default="data/test_images/bus.jpg")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save-dir", default="reports")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    agent = InspectionAgent()
    state = agent.run(args.image, conf_thres=args.conf,
                      save=not args.no_save, save_dir=args.save_dir)

    print(f"\n{'='*60}")
    print("  Agent Summary")
    print(f"{'='*60}")
    print(f"Image:       {state['image_path']}")
    print(f"Objects:     {state['det_result']['counts']}")
    print(f"Risk level:  {state.get('risk_level', 'n/a')}")
    print(f"Standards:   {len(state.get('standards', []))} retrieved")
    if state.get("decisions"):
        print("Decisions:")
        for d in state["decisions"]:
            print(f"  - {d}")
