"""Session Replay — pick a session, scrub through every tick + decision."""
from __future__ import annotations

import streamlit as st


def main() -> None:
    st.title("Session Replay")
    st.caption("Tick-by-tick replay of a session with risk decisions and fills")
    session_id = st.text_input("Session ID (UUID)")
    if session_id:
        st.info(f"Session {session_id}: wire to session repository and audit log.")


if __name__ == "__main__":
    main()
