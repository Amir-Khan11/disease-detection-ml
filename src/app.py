from __future__ import annotations

from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import difflib
import os
import re
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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
except ImportError as exc:
    np = None
    pd = None
    plt = None
    KMeans = None
    PCA = None
    RandomForestClassifier = None
    accuracy_score = None
    confusion_matrix = None
    train_test_split = None
    DecisionTreeClassifier = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = None


# Everyday phrases -> dataset symptom columns (used only in real-dataset mode; unknown columns are ignored)
REAL_SYNONYMS = {
    "fever": ["high_fever"], "temperature": ["high_fever"], "high temperature": ["high_fever"],
    "chill": ["chills"], "shaking": ["shivering"],
    "body pain": ["muscle_pain", "joint_pain"], "body ache": ["muscle_pain", "joint_pain"],
    "muscle ache": ["muscle_pain"], "joint ache": ["joint_pain"], "back ache": ["back_pain"],
    "head ache": ["headache"], "head pain": ["headache"],
    "throwing up": ["vomiting"], "vomit": ["vomiting"],
    "tired": ["fatigue"], "tiredness": ["fatigue"], "exhausted": ["fatigue"],
    "weakness": ["muscle_weakness", "fatigue"],
    "rash": ["skin_rash"], "red spots": ["red_spots_over_body"], "itch": ["itching"],
    "sore throat": ["throat_irritation", "patches_in_throat"], "throat pain": ["throat_irritation"],
    "stuffy nose": ["congestion"], "blocked nose": ["congestion"],
    "short of breath": ["breathlessness"], "shortness of breath": ["breathlessness"],
    "difficulty breathing": ["breathlessness"],
    "stomach ache": ["stomach_pain"], "tummy pain": ["stomach_pain"], "belly pain": ["abdominal_pain"],
    "loose motion": ["diarrhoea"], "diarrhea": ["diarrhoea"],
    "dizzy": ["dizziness"], "sweat": ["sweating"],
    "yellow skin": ["yellowish_skin"], "yellow eyes": ["yellowing_of_eyes"],
    "painful urination": ["burning_micturition"], "burning urination": ["burning_micturition"],
    "frequent urination": ["polyuria"],
}


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

        self.model_name = "Decision Tree"
        self.dataset_name = "Synthetic Dataset"
        self.real_data_path = self.base_dir / "data" / "real" / "symptoms_binary.csv"
        self.use_real_data = self.real_data_path.exists()
        if self.use_real_data:
            self._setup_real_data()

        self.dataset = self.real_df if self.use_real_data else self._build_dataset()
        self.model, self.report = self._train_supervised_model()
        self.cluster_model, self.pca_model, self.cluster_labels = self._train_unsupervised_model()
        self._generate_visual_reports()

    @staticmethod
    def _norm(text: object) -> str:
        return re.sub(r"[^a-z0-9]", "", str(text).lower())

    @staticmethod
    def _display(text: object) -> str:
        return " ".join(str(text).replace("_", " ").split())

    def _lookup(self, key: str, mapping: dict):
        if key in mapping:
            return mapping[key]
        close = difflib.get_close_matches(key, list(mapping), n=1, cutoff=0.85)
        return mapping[close[0]] if close else None

    def _setup_real_data(self) -> None:
        """Use the real symptom dataset (data/real/) with Random Forest, descriptions and precautions."""
        data_dir = self.real_data_path.parent
        df = pd.read_csv(self.real_data_path).drop_duplicates().reset_index(drop=True)
        df["disease"] = df["disease"].map(self._display)
        symptoms = [column for column in df.columns if column != "disease"]

        self.real_df = df
        self.symptom_order = symptoms
        self.symptom_map = {symptom: [self._display(symptom)] for symptom in symptoms}
        for phrase, columns in REAL_SYNONYMS.items():
            for column in columns:
                if column in self.symptom_map:
                    self.symptom_map[column].append(phrase)

        descriptions: dict[str, str] = {}
        description_file = data_dir / "symptom_Description.csv"
        if description_file.exists():
            for _, row in pd.read_csv(description_file).iterrows():
                descriptions[self._norm(row.iloc[0])] = str(row.iloc[1]).strip()

        precautions: dict[str, list[str]] = {}
        precaution_file = data_dir / "symptom_precaution.csv"
        if precaution_file.exists():
            for _, row in pd.read_csv(precaution_file).iterrows():
                items = [str(item).strip().capitalize() for item in row.iloc[1:].dropna() if str(item).strip()]
                precautions[self._norm(row.iloc[0])] = items

        diseases = sorted(df["disease"].unique())
        self.disease_profiles = {disease: None for disease in diseases}
        self.advice = {}
        self.severity_notes = {}
        for disease in diseases:
            key = self._norm(disease)
            steps = self._lookup(key, precautions)
            about = self._lookup(key, descriptions)
            self.advice[disease] = (
                "Suggested precautions: " + "; ".join(steps) + ". Please consult a doctor for a proper diagnosis."
                if steps else "Please consult a doctor for a proper diagnosis."
            )
            self.severity_notes[disease] = f"About {disease}: {about}" if about else "No description available."

        self.model_name = "Random Forest"
        self.dataset_name = "Real Dataset (Kaggle Disease Symptom Prediction)"

    def _term_in_text(self, term: str, text: str) -> bool:
        if self.use_real_data:
            return re.search(r"\b" + re.escape(term), text) is not None
        return term in text

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

        if self.use_real_data:
            test_size = max(0.2, (y_values.nunique() + 1) / len(y_values))
            stratify = y_values if y_values.value_counts().min() >= 2 else None
            model = RandomForestClassifier(n_estimators=200, random_state=42)
        else:
            test_size = 0.25
            stratify = y_values
            model = DecisionTreeClassifier(max_depth=6, random_state=42)

        x_train, x_test, y_train, y_test = train_test_split(
            x_values,
            y_values,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )

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
        many = len(counts) > 12
        plt.figure(figsize=(14, 7) if many else (10, 5))
        plt.bar(counts.index, counts.values, color="#10a37f")
        plt.title(f"Disease Distribution in {self.dataset_name}")
        plt.ylabel("Sample Count")
        plt.xticks(rotation=90 if many else 20, fontsize=7 if many else 10)
        plt.tight_layout()
        plt.savefig(self.output_dir / "class_distribution.png")
        plt.close()

        labels = sorted(self.dataset["disease"].unique())
        matrix = confusion_matrix(self.test_labels, self.test_predictions, labels=labels)
        many = len(labels) > 15
        plt.figure(figsize=(14, 12) if many else (8, 6))
        plt.imshow(matrix, cmap="Blues")
        plt.title(f"{self.model_name} Confusion Matrix")
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=90 if many else 35, ha="center" if many else "right", fontsize=6 if many else 10)
        plt.yticks(range(len(labels)), labels, fontsize=6 if many else 10)
        if not many:
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
            found = any(self._term_in_text(term, lowered) for term in self.symptom_map[symptom])
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
                advice=self.advice.get("healthy", ""),
                severity=self.severity_notes.get("healthy", ""),
                cluster_label=0,
            )

        probabilities = self.model.predict_proba(input_frame)[0]
        predictions = list(zip(self.model.classes_, probabilities))
        predictions.sort(key=lambda item: item[1], reverse=True)
        top_label = predictions[0][0]

        return DiagnosisResult(
            recognized_symptoms=recognized,
            predictions=predictions[:3],
            advice=self.advice.get(top_label, ""),
            severity=self.severity_notes.get(top_label, ""),
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
            "model": self.model_name,
            "dataset": self.dataset_name,
            "dataset_path": str(
                self.real_data_path if self.use_real_data
                else self.base_dir / "data" / "synthetic" / "synthetic_symptom_dataset.csv"
            ),
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
