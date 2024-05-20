### Benchmarks, groups and tasks
- how does it infer tasks for --tasks polish?
file: lm_eval/tasks/__init__.py#L105
{'type': 'group', 'task': -1, 'yaml_path': '/home/mikolaj/github/lm-evaluation-harness/lm_eval/tasks/benchmarks/polish.yaml'}

- task can be called be specyfing the yaml path
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md
--tasks /path/to/yaml/file

