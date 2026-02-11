# Contributing Guide

## 1. Create or Pick an Issue

- Every task starts with an Issue.
- Assign one owner and target date.

## 2. Branch Naming

- `feature/<task-name>`
- `fix/<task-name>`
- `docs/<task-name>`

Examples:

- `feature/clinvar-cleaning`
- `feature/cnn-training`
- `fix/label-leakage`

## 3. Commit Message Format

Use short conventional commits:

- `feat: add clinvar preprocessing pipeline`
- `fix: correct pathogenic label mapping`
- `docs: update training instructions`
- `chore: add data folder placeholders`

## 4. Pull Request Rules

- PR target is `develop` (not `main`).
- At least 1 reviewer approval (recommended 2 for core logic).
- CI/checks must pass before merge.
- Link issue in PR description: `Closes #12`.

## 5. Merge Rules

- Use squash merge for feature branches.
- Delete merged branches.
- `main` should only receive stable tested releases from `develop`.

