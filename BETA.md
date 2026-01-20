# Beta Release Policy

This repository is currently in **Beta**.

## What "Beta" Means

A beta release indicates that the project is usable and intentionally published, but **not yet stable**. The core ideas and direction are being validated, and feedback is encouraged.

Specifically:

* APIs and public interfaces **may change without notice**
* Breaking changes can occur between releases
* Bugs and incomplete behavior are expected
* No backward-compatibility guarantees are made

## Beta Tags and Releases

Tags or releases named `beta` (or containing `-beta`) are:

* Intended for **early adopters and testers**
* Suitable for experimentation and evaluation
* **Not recommended for production use**

The `beta` tag may point to an early milestone and should not be relied on as a long-term pinned dependency.

## How to Use Beta Safely

If you choose to use a beta release:

* Pin to a specific commit SHA if stability matters
* Expect to update your code as the project evolves
* Follow the changelog and release notes closely

## Path to Stable

The project will move toward versioned releases (for example `v0.x` or `v1.0.0`) once:

* Core APIs are finalized
* Usage patterns are validated
* Breaking changes become rare and intentional

Until then, all releases should be considered **beta-quality**.

---

Feedback, issues, and discussions are highly welcome during this phase.
