# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues. Use the `gh` CLI.

- Create: `gh issue create --title "..." --body "..."`
- Read:   `gh issue view <number> --comments`
- List:   `gh issue list --state open --json number,title,body,labels`
- Close:  `gh issue close <number> --comment "..."`
- Labels: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`

Infer the repo from `git remote -v` — `gh` does this automatically.
