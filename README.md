# Genetic Mutation Detection (Team Workspace)

This repository is configured for a 6-person graduation project team to collaborate safely using GitHub pull requests.

## Branch Strategy

- `main`: production-ready code only.
- `develop`: integration branch for completed features.
- `feature/<short-task-name>`: each task branch created from `develop`.
- `fix/<short-task-name>`: bug fix branch created from `develop`.

## Daily Workflow

1. Update local `develop`.
2. Create your feature branch.
3. Commit in small logical steps.
4. Push branch and open a PR to `develop`.
5. Merge only after review and checks pass.

## First-Time Setup

```bash
git checkout -b develop
git add .
git commit -m "chore: initialize team collaboration structure"
```

After you create the GitHub repository and add it as `origin`:

```bash
git push -u origin main
git push -u origin develop
```

## Suggested Folder Structure

```text
data/
  raw/
  processed/
notebooks/
src/
  data/
  features/
  models/
  evaluation/
reports/
```

## Team Rules

- No direct push to `main`.
- No merge without review.
- Every PR must link an Issue.
- Keep dataset files out of git unless small metadata files.

