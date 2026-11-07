# LlamaNX Workspace Guidelines

## Release & Version Invariants
- **Version Bumping:** When building a new release, bump `APP_VERSION` in the `Makefile` and make a new git commit before tagging.
- **Asset Naming Invariant:** The compiled `.nro` uploaded to GitHub MUST match the release version:
  - Release `v1.1.0` -> `LlamaNX-v1.1.0.nro`
  - Release `v1.1.1` -> `LlamaNX-v1.1.1.nro`
  - Release `v1.1.2` -> `LlamaNX-v1.1.2.nro`
  - Release `v1.1.3` -> `LlamaNX-v1.1.3.nro`
- **Release Formatting:**
  - Tag: `vX.Y.Z`
  - Name: `LlamaNX vX.Y.Z - <Feature Summary>`
  - Body: Begins with `## LlamaNX vX.Y.Z`
- **Strict Immutability:** Never overwrite or re-upload files to an existing release tag. Always create a new release for new builds.
- **No Overwriting:** Every change must be committed to git as a distinct new commit.
