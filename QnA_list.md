
### 2026-11-05 (Thu) 14:04
**질문/지시:** TG-CTRG 로 정함. `VLM3D_MICCAI/TG-CTRG` 에 release 내용을 옮기고 GitHub 연동함. 모델 이름을 TG-CTRG 로 바꿔줘.

**답:** `TG-CTRG/README.md` 제목 → "TG-CTRG: Template-Guided CT Report Generation", bibtex 키/제목 변경, `<GITHUB_REPO_URL>` → `https://github.com/babbu3682/TG-CTRG`(커밋 author 의 GitHub 계정에서 추정; 원격은 아직 미설정), `<HF_REPO>` → `<HF_USER>/TG-CTRG`. `CITATION.cff` 제목·URL·license(MIT) 갱신. `hf_weights/README.md` 제목·URL 갱신. GitHub Desktop 이 추가한 루트 `LICENSE`(MIT)와 CITATION 의 CC-BY-NC-SA 가 충돌해 코드 MIT / 서브트리 ECL·Apache 유지 / 가중치 CC BY-NC-SA 로 README 에 License 절 추가. `release/` 는 사라졌으므로(이동) 빌드 도구는 `TG-CTRG/_private/tools/`. 커밋은 안 함(사용자가 Desktop 에서).
