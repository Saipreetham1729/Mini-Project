import { Component } from "react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, info) {
    // eslint-disable-next-line no-console
    console.error("ContextGuard UI error:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="card">
          <h3>Something went wrong on this page</h3>
          <p>
            This is a display error, not lost data -- your experiment results are still intact.
            Reloading this page usually fixes it.
          </p>
          <button className="btn" onClick={() => window.location.reload()}>
            Reload
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
