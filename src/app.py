from __future__ import annotations

from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import os
import webbrowser

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

try:
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
except ImportError as exc:
    np = None
    pd = None
    plt = None
    KMeans = None
    PCA = None
    accuracy_score = None
    confusion_matrix = None
    train_test_split = None
    DecisionTreeClassifier = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = None


@dataclass
class DiagnosisResult:
    recognized_symptoms: list[str]
    predictions: list[tuple[str, float]]
    advice: str
    severity: str
    cluster_label: int


@dataclass
class MLReport:
    dataset_rows: int
    train_rows: int
    test_rows: int
    accuracy: float
    cluster_count: int
    disease_counts: dict[str, int]


class DiseaseDetectionML:
    def __init__(self) -> None:
        if IMPORT_ERROR:
            raise RuntimeError(
                "Required packages are missing. Run: pip install -r requirements.txt. "
                f"Import error: {IMPORT_ERROR}"
            )

        self.base_dir = Path(__file__).resolve().parent.parent
        self.web_dir = self.base_dir / "webapp"
        self.output_dir = self.base_dir / "reports"
        self.output_dir.mkdir(exist_ok=True)

        self.symptom_order = [
            "fever",
            "headache",
            "vomiting",
            "chills",
            "cough",
            "fatigue",
            "body_pain",
            "rash",
            "sore_throat",
        ]

        self.symptom_map = {
            "fever": ["fever", "high temperature", "temperature", "hot body"],
            "headache": ["headache", "head pain", "pain in head"],
            "vomiting": ["vomiting", "nausea", "throwing up"],
            "chills": ["chills", "chill", "shivering"],
            "cough": ["cough", "coughing", "dry cough"],
            "fatigue": ["fatigue", "weakness", "tired", "exhausted"],
            "body_pain": ["body pain", "body ache", "muscle pain", "joint pain"],
            "rash": ["rash", "skin rash", "red spots"],
            "sore_throat": ["sore throat", "throat pain", "painful throat"],
        }

        self.disease_profiles = {
            "malaria": [1, 1, 1, 1, 0, 1, 1, 0, 0],
            "dengue": [1, 1, 1, 0, 0, 1, 1, 1, 0],
            "viral fever": [1, 0, 0, 0, 1, 1, 1, 0, 1],
            "flu": [1, 1, 0, 0, 1, 1, 1, 0, 1],
            "common cold": [0, 1, 0, 0, 1, 0, 0, 0, 1],
            "typhoid": [1, 1, 1, 0, 0, 1, 1, 0, 0],
            "healthy": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        }

        self.advice = {
            "malaria": "Rest, drink fluids, and get proper medical testing.",
            "dengue": "Drink plenty of water and seek professional medical advice quickly.",
            "viral fever": "Rest well, monitor temperature, and stay hydrated.",
            "flu": "Rest, fluids, and monitor breathing or worsening fever.",
            "common cold": "Usually mild. Rest and warm fluids can help.",
            "typhoid": "Medical consultation is recommended for proper treatment.",
            "healthy": "No major disease pattern is visible in the entered symptoms.",
        }

        self.severity_notes = {
            "malaria": "Moderate risk pattern. Lab testing is strongly recommended.",
            "dengue": "Moderate to high risk pattern. Please seek medical evaluation.",
            "viral fever": "Usually mild to moderate, but keep monitoring symptoms.",
            "flu": "Usually manageable, but watch breathing and persistent fever.",
            "common cold": "Low risk pattern based on current symptoms.",
            "typhoid": "Moderate risk pattern. Medical attention is recommended.",
            "healthy": "Very low risk pattern based on current inputs.",
        }

        self.dataset = self._build_dataset()
        self.model, self.report = self._train_supervised_model()
        self.cluster_model, self.pca_model, self.cluster_labels = self._train_unsupervised_model()
        self._generate_visual_reports()

    def _build_dataset(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        rows: list[list[object]] = []

        for disease, profile in self.disease_profiles.items():
            sample_count = 60 if disease != "healthy" else 30
            base = np.array(profile)

            for _ in range(sample_count):
                row = base.copy()
                flip_indices = rng.choice(len(base), size=rng.integers(0, 3), replace=False)
                for index in flip_indices:
                    flip_probability = 0.18 if disease != "healthy" else 0.10
                    if rng.random() < flip_probability:
                        row[index] = 1 - row[index]
                rows.append(list(row) + [disease])

        df = pd.DataFrame(rows, columns=self.symptom_order + ["disease"])
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df.to_csv(self.base_dir / "data" / "synthetic" / "synthetic_symptom_dataset.csv", index=False)
        return df

    def _train_supervised_model(self) -> tuple[DecisionTreeClassifier, MLReport]:
        x_values = self.dataset[self.symptom_order]
        y_values = self.dataset["disease"]

        x_train, x_test, y_train, y_test = train_test_split(
            x_values,
            y_values,
            test_size=0.25,
            random_state=42,
            stratify=y_values,
        )

        model = DecisionTreeClassifier(max_depth=6, random_state=42)
        model.fit(x_train, y_train)

        self.test_labels = y_test
        self.test_predictions = model.predict(x_test)

        report = MLReport(
            dataset_rows=len(self.dataset),
            train_rows=len(x_train),
            test_rows=len(x_test),
            accuracy=float(accuracy_score(y_test, self.test_predictions)),
            cluster_count=len(self.disease_profiles),
            disease_counts=self.dataset["disease"].value_counts().sort_index().to_dict(),
        )
        return model, report

    def _train_unsupervised_model(self) -> tuple[KMeans, PCA, np.ndarray]:
        features = self.dataset[self.symptom_order]
        cluster_count = len(self.disease_profiles)
        cluster_model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        cluster_labels = cluster_model.fit_predict(features)

        pca_model = PCA(n_components=2, random_state=42)
        self.pca_points = pca_model.fit_transform(features)
        return cluster_model, pca_model, cluster_labels

    def _generate_visual_reports(self) -> None:
        counts = self.dataset["disease"].value_counts().sort_values(ascending=False)
        plt.figure(figsize=(10, 5))
        plt.bar(counts.index, counts.values, color="#10a37f")
        plt.title("Disease Distribution in Synthetic Dataset")
        plt.ylabel("Sample Count")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / "class_distribution.png")
        plt.close()

        labels = sorted(self.dataset["disease"].unique())
        matrix = confusion_matrix(self.test_labels, self.test_predictions, labels=labels)
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap="Blues")
        plt.title("Decision Tree Confusion Matrix")
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
        plt.yticks(range(len(labels)), labels)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color="#111827")
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(
            self.pca_points[:, 0],
            self.pca_points[:, 1],
            c=self.cluster_labels,
            cmap="tab10",
            alpha=0.8,
        )
        plt.title("KMeans Clusters of Symptom Patterns")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.tight_layout()
        plt.savefig(self.output_dir / "symptom_clusters.png")
        plt.close()

    def parse_symptoms(self, user_text: str) -> tuple[pd.DataFrame, list[str]]:
        lowered = user_text.lower()
        vector = []
        recognized = []

        for symptom in self.symptom_order:
            found = any(term in lowered for term in self.symptom_map[symptom])
            vector.append(1 if found else 0)
            if found:
                recognized.append(symptom.replace("_", " "))

        return pd.DataFrame([vector], columns=self.symptom_order), recognized

    def diagnose(self, user_text: str) -> DiagnosisResult:
        input_frame, recognized = self.parse_symptoms(user_text)
        if not recognized:
            return DiagnosisResult(
                recognized_symptoms=[],
                predictions=[("healthy", 1.0)],
                advice=self.advice["healthy"],
                severity=self.severity_notes["healthy"],
                cluster_label=0,
            )

        probabilities = self.model.predict_proba(input_frame)[0]
        predictions = list(zip(self.model.classes_, probabilities))
        predictions.sort(key=lambda item: item[1], reverse=True)
        top_label = predictions[0][0]

        return DiagnosisResult(
            recognized_symptoms=recognized,
            predictions=predictions[:3],
            advice=self.advice[top_label],
            severity=self.severity_notes[top_label],
            cluster_label=int(self.cluster_model.predict(input_frame)[0]),
        )

    def summary_payload(self) -> dict[str, object]:
        return {
            "dataset_rows": self.report.dataset_rows,
            "train_rows": self.report.train_rows,
            "test_rows": self.report.test_rows,
            "accuracy": round(self.report.accuracy * 100, 2),
            "cluster_count": self.report.cluster_count,
            "disease_counts": self.report.disease_counts,
            "plots": {
                "class_distribution": "/plots/class_distribution.png",
                "confusion_matrix": "/plots/confusion_matrix.png",
                "symptom_clusters": "/plots/symptom_clusters.png",
            },
            "dataset_path": str(self.base_dir / "data" / "synthetic" / "synthetic_symptom_dataset.csv"),
        }


class DiseaseRequestHandler(SimpleHTTPRequestHandler):
    engine: DiseaseDetectionML | None = None
    web_root: Path | None = None
    output_root: Path | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.web_root), **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/summary":
            self._send_json(self.engine.summary_payload())
            return

        if self.path.startswith("/plots/"):
            filename = Path(self.path.removeprefix("/plots/")).name
            file_path = self.output_root / filename
            if file_path.exists():
                return SimpleHTTPRequestHandler.do_GET(self)
            self.send_error(HTTPStatus.NOT_FOUND, "Plot not found")
            return

        if self.path == "/":
            self.path = "/index.html"
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self) -> None:
        if self.path != "/api/diagnose":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
            return

        user_text = str(payload.get("message", "")).strip()
        if not user_text:
            self._send_json({"error": "Please enter symptoms first."}, status=HTTPStatus.BAD_REQUEST)
            return

        result = self.engine.diagnose(user_text)
        response = asdict(result)
        response["predictions"] = [
            {"disease": disease, "probability": round(probability * 100, 2)}
            for disease, probability in result.predictions
        ]
        self._send_json(response)

    def translate_path(self, path: str) -> str:
        if path.startswith("/plots/"):
            filename = Path(path.removeprefix("/plots/")).name
            return str(self.output_root / filename)
        return super().translate_path(path)

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_cli() -> None:
    engine = DiseaseDetectionML()
    print("Disease Detection ML Assistant")
    print("Type symptoms in plain English. Example: fever, chills, body pain")

    user_text = input("\nEnter your symptoms: ").strip()
    result = engine.diagnose(user_text)

    print("\nRecognized symptoms:")
    print(", ".join(result.recognized_symptoms) if result.recognized_symptoms else "None")

    print("\nTop predictions:")
    for disease, probability in result.predictions:
        print(f"{disease.title()} : {probability * 100:.2f}%")

    print("\nAdvice:")
    print(result.advice)

    print("\nSeverity note:")
    print(result.severity)

    print(f"\nKMeans cluster: Cluster {result.cluster_label}")
    print(f"\nAccuracy: {engine.report.accuracy * 100:.2f}%")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Disease Detection ML web app")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local web server")
    parser.add_argument("--port", type=int, default=8000, help="Port for the local web server")
    parser.add_argument("--open-browser", action="store_true", help="Open the app in your browser on startup")
    parser.add_argument("--cli", metavar="SYMPTOMS", help="Run a one-off CLI diagnosis instead of starting the web server")
    return parser


def run_cli_message(symptoms: str) -> None:
    engine = DiseaseDetectionML()
    result = engine.diagnose(symptoms)
    print(json.dumps({
        "recognized_symptoms": result.recognized_symptoms,
        "predictions": [
            {"disease": disease, "probability": round(probability * 100, 2)}
            for disease, probability in result.predictions
        ],
        "advice": result.advice,
        "severity": result.severity,
        "cluster_label": result.cluster_label,
        "accuracy": round(engine.report.accuracy * 100, 2),
    }, indent=2))


def run_web_app(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = False) -> None:
    engine = DiseaseDetectionML()

    DiseaseRequestHandler.engine = engine
    DiseaseRequestHandler.web_root = engine.web_dir
    DiseaseRequestHandler.output_root = engine.output_dir

    server = ThreadingHTTPServer((host, port), DiseaseRequestHandler)
    url = f"http://{host}:{port}"

    print(f"Server running at {url}")
    print("Press Ctrl+C to stop the server.")

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


def main() -> None:
    if IMPORT_ERROR:
        print(
            "Missing required packages. Run: pip install -r requirements.txt\n\n"
            f"Import error: {IMPORT_ERROR}"
        )
        return

    args = build_parser().parse_args()
    if args.cli:
        run_cli_message(args.cli)
        return

    run_web_app(host=args.host, port=args.port, open_browser=args.open_browser)


if __name__ == "__main__":
    main()
