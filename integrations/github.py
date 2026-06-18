# -*- coding: utf-8 -*-
"""integrations/github.py — GitHub REST API integration."""
from __future__ import annotations
import logging
import os
from typing import Any

log = logging.getLogger("integrations.github")


class GitHubIntegration:
    def __init__(self, token: str = ""):
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self._base = "https://api.github.com"

    def _headers(self) -> dict:
        h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _get(self, path: str) -> Any:
        import httpx
        r = httpx.get(f"{self._base}{path}", headers=self._headers(), timeout=15)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: dict) -> Any:
        import httpx
        r = httpx.post(f"{self._base}{path}", json=data, headers=self._headers(), timeout=15)
        r.raise_for_status()
        return r.json()

    def get_repo(self, owner: str, repo: str) -> dict:
        return self._get(f"/repos/{owner}/{repo}")

    def list_issues(self, owner: str, repo: str, state: str = "open") -> list[dict]:
        return self._get(f"/repos/{owner}/{repo}/issues?state={state}&per_page=50")

    def create_issue(self, owner: str, repo: str, title: str, body: str = "",
                     labels: list[str] | None = None) -> dict:
        payload: dict = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels
        return self._post(f"/repos/{owner}/{repo}/issues", payload)

    def create_pr(self, owner: str, repo: str, title: str, head: str, base: str,
                  body: str = "") -> dict:
        return self._post(f"/repos/{owner}/{repo}/pulls",
                          {"title": title, "head": head, "base": base, "body": body})

    def list_prs(self, owner: str, repo: str, state: str = "open") -> list[dict]:
        return self._get(f"/repos/{owner}/{repo}/pulls?state={state}&per_page=50")

    def get_pr_comments(self, owner: str, repo: str, pr_number: int) -> list[dict]:
        return self._get(f"/repos/{owner}/{repo}/issues/{pr_number}/comments")

    def add_pr_comment(self, owner: str, repo: str, pr_number: int, body: str) -> dict:
        return self._post(f"/repos/{owner}/{repo}/issues/{pr_number}/comments", {"body": body})

    def list_workflows(self, owner: str, repo: str) -> list[dict]:
        data = self._get(f"/repos/{owner}/{repo}/actions/workflows")
        return data.get("workflows", [])

    def trigger_workflow(self, owner: str, repo: str, workflow_id: str, ref: str = "main",
                         inputs: dict | None = None) -> bool:
        import httpx
        r = httpx.post(
            f"{self._base}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches",
            json={"ref": ref, "inputs": inputs or {}},
            headers=self._headers(), timeout=15
        )
        return r.status_code == 204

    def setup_webhook(self, owner: str, repo: str, url: str,
                      events: list[str] | None = None) -> dict:
        return self._post(f"/repos/{owner}/{repo}/hooks", {
            "name": "web",
            "active": True,
            "events": events or ["push", "pull_request"],
            "config": {"url": url, "content_type": "json"},
        })
