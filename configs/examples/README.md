# Example Configurations

The files in this directory are reference examples only. They document real
configuration patterns and integrations, but they are not guaranteed to work
on your machine without editing.

Create your actual machine-specific configurations in `configs/local/`.
That directory is intentionally ignored by Git, except for its keep-file and
sanitization guide, so local endpoints, hardware details, private paths,
credentials, and other environment-specific values stay out of version
control.

The example configurations may include known hardware and operating-system
details to make the settings concrete. Treat those values as illustrative,
not as project defaults or benchmark claims. Review every field before using
an example, especially model paths, URLs, commands, evaluator integrations,
and output paths.

The example evaluator script is provided as a sample integration. It may
require additional software, network access, and its own security review.

Before adapting an engine- or model-specific example, review the corresponding
profile under `docs/hosting/`. Those profiles distinguish portable settings
from operating-system, accelerator, and machine-specific observations.