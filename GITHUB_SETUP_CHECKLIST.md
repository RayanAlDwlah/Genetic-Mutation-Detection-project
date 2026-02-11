# GitHub Team Setup Checklist

Use this after creating the remote repository on GitHub.

## 1. Add Team Members

- Repository `Settings` -> `Collaborators and teams`.
- Add all 6 members with `Write` access.

## 2. Protect `main`

Repository `Settings` -> `Branches` -> `Add branch protection rule`:

- Branch name pattern: `main`
- Require a pull request before merging
- Require approvals: `2`
- Dismiss stale approvals when new commits are pushed
- Require conversation resolution before merging
- Include administrators
- Disable force pushes
- Disable deletions

## 3. Protect `develop`

Add another rule for `develop`:

- Require a pull request before merging
- Require approvals: `1`
- Require conversation resolution before merging
- Disable force pushes

## 4. Optional but Recommended

- `Settings` -> `General` -> enable auto-delete head branches.
- `Settings` -> `Actions` -> allow GitHub Actions.
- `Projects` -> create board with columns:
  - `Backlog`
  - `In Progress`
  - `Review`
  - `Done`

## 5. Team Working Agreement

- No direct pushes to `main`.
- All code through PRs.
- One issue per task.
- One owner per issue.

