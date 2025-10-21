<!--
Copyright (C) 2025 Postquant Labs Incorporated
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Contributing to Portfolio Optimization (QPO)

Thank you for your interest in contributing to the Portfolio Optimization project! This document provides guidelines and information about contributing to this AGPLv3-licensed project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [License Compliance](#license-compliance)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Submitting Contributions](#submitting-contributions)
- [Contributor License Agreement](#contributor-license-agreement)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## License Compliance

### Understanding the AGPL-3.0 License

This project is licensed under the **GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)**. By contributing to this project, you agree that your contributions will be licensed under the same license.

#### Key Points:

1. **Source Code Availability**: Any modifications to this software that are provided as a network service must make the complete source code available to users.

2. **Copyleft**: The AGPL is a strong copyleft license. Any derivative works must also be licensed under AGPL-3.0 (or a compatible license).

3. **Patent Grant**: Contributors grant a patent license for their contributions as specified in the AGPL-3.0.

### Adding License Headers to New Files

**ALL new source code files MUST include the following license header:**

#### For Python files (.py):

```python
# Copyright (C) 2025 Postquant Labs Incorporated
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
```

#### For Markdown files (.md):

```markdown
<!--
Copyright (C) 2025 Postquant Labs Incorporated
SPDX-License-Identifier: AGPL-3.0-or-later
-->
```

#### For other file types:

Adapt the comment syntax appropriately while keeping the same content.

### Third-Party Code and Dependencies

#### Adding New Dependencies

When adding a new third-party dependency:

1. **Check License Compatibility**: Ensure the dependency's license is compatible with AGPL-3.0. Compatible licenses include:
   - **Permissive licenses**: MIT, BSD (2-clause, 3-clause), Apache-2.0, ISC
   - **Copyleft licenses**: GPL-3.0, LGPL-3.0
   - **Incompatible**: Proprietary licenses, some older GPL versions (GPL-2.0-only without "or later" clause)

2. **Update NOTICE File**: If the dependency is Apache-2.0 licensed or requires attribution:
   - Add copyright and license information to the `NOTICE` file
   - Check if the dependency includes its own NOTICE file and incorporate required attributions

3. **Update requirements.txt**: Add the dependency with appropriate version constraints

4. **Document the Dependency**: Update README.md if the dependency is significant

#### Incorporating Third-Party Code

If you incorporate code from other projects:

1. **Verify License Compatibility**: Ensure the code's license is AGPL-3.0 compatible
2. **Preserve Original Copyright**: Keep the original copyright notice and license header
3. **Document the Source**: Add a comment indicating the original source, author, and license
4. **Update NOTICE**: Add appropriate attribution to the NOTICE file
5. **Dual Headers**: If needed, include both the original license header and a note about AGPL-3.0 licensing

Example:
```python
# Original code from [Project Name]
# Copyright (C) [Year] [Original Author]
# Licensed under [Original License]
#
# Modified for Portfolio Optimization (QPO)
# Copyright (C) 2025 Postquant Labs Incorporated
#
# This modified version is licensed under AGPL-3.0-or-later
# [Full AGPL header...]
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv or virtualenv)

### Setting Up Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/yourusername/portfolio-optimization.git
   cd portfolio-optimization
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Run tests** to ensure everything works:
   ```bash
   pytest tests/
   ```

## Development Workflow

### Branch Naming

Use descriptive branch names:
- `feature/description` - For new features
- `fix/description` - For bug fixes
- `docs/description` - For documentation updates
- `refactor/description` - For code refactoring

### Commit Messages

Write clear, descriptive commit messages:

```
Short summary (50 chars or less)

More detailed explanation if necessary. Wrap at 72 characters.
Explain the problem this commit solves and why this approach was chosen.

Fixes #123
```

### Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage

### Code Review

All contributions must go through code review:
1. Submit a pull request with a clear description
2. Address reviewer feedback promptly
3. Ensure CI/CD checks pass
4. Maintain respectful and constructive communication

## Coding Standards

### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **type hints** for function arguments and return values
- Write **docstrings** for all public modules, functions, classes, and methods
- Keep functions focused and small
- Use meaningful variable and function names

Example:
```python
def calculate_portfolio_return(
    weights: np.ndarray,
    returns: pd.DataFrame
) -> float:
    """
    Calculate the expected portfolio return.

    Args:
        weights: Portfolio weights as a numpy array
        returns: Historical returns as a pandas DataFrame

    Returns:
        Expected portfolio return as a float
    """
    # Implementation here
```

### Documentation

- Update relevant documentation when making changes
- Include examples for new features
- Keep README.md up to date
- Document any breaking changes

## Submitting Contributions

### Pull Request Process

1. **Ensure your code follows the guidelines** in this document
2. **Add the license header** to any new files
3. **Update documentation** as needed
4. **Write or update tests** for your changes
5. **Run the test suite** and ensure all tests pass
6. **Update CHANGELOG** (if applicable) with your changes
7. **Submit a pull request** with a clear description

### Pull Request Description Template

```markdown
## Description
[Describe what this PR does]

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
[Describe the tests you ran and their results]

## License Compliance Checklist
- [ ] All new files include the required AGPL-3.0 license header
- [ ] Any new dependencies are AGPL-3.0 compatible
- [ ] NOTICE file updated if adding Apache-2.0 licensed dependencies
- [ ] No proprietary or incompatible code was incorporated

## Additional Notes
[Any additional information reviewers should know]
```

## Contributor License Agreement

By submitting a contribution to this project, you:

1. **Certify** that you wrote the contribution or have the right to submit it under the AGPL-3.0 license
2. **Agree** that your contribution will be licensed under AGPL-3.0-or-later
3. **Grant** a patent license as specified in the AGPL-3.0 license
4. **Acknowledge** that your contribution is public and may be redistributed under the AGPL-3.0 license

This is based on the [Developer Certificate of Origin (DCO)](https://developercertificate.org/):

```
Developer Certificate of Origin
Version 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the AGPL-3.0 license; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the AGPL-3.0 license; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the AGPL-3.0 license.
```

### Sign-off Procedure

Add a `Signed-off-by` line to your commit messages:

```bash
git commit -s -m "Your commit message"
```

This adds:
```
Signed-off-by: Your Name <your.email@example.com>
```

## Questions?

If you have questions about contributing, licensing, or anything else, please:

1. Check existing documentation and issues
2. Open a new issue with your question
3. Reach out to the maintainers

Thank you for contributing to Portfolio Optimization (QPO)!

---

**License**: This document is licensed under AGPL-3.0-or-later
**Copyright**: (C) 2025 Postquant Labs Incorporated
