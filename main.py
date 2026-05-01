import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    # Lazy import so .env is loaded before OpenAI client initialises
    from agent.orchestrator import CodingAgent

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        print("╔══════════════════════════════════════╗")
        print("║     CODING AGENT  — powered by GPT   ║")
        print("╚══════════════════════════════════════╝")
        task = input("\nEnter your coding task: ").strip()
        if not task:
            print("No task provided. Exiting.")
            return

    agent = CodingAgent()
    agent.run(task)


if __name__ == "__main__":
    main()
