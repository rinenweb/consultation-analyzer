TRANSLATIONS = {

    "en": {

        # --- Header ---
        "title": "Public Consultation Analysis",
        "subtitle": "Rule-based transparency tool for consultation analysis.",

        # --- Input ---
        "input_label": "Enter consultation URL",
        "run": "Run Analysis",

        # --- Advanced ---
        "advanced": "Advanced Settings",
        "policy": "Policy keywords (comma separated)",
        "amend": "Amendment verbs (comma separated)",
        # --- Advanced tooltips ---
        "policy_help": (
            "Comma-separated keyword fragments used to detect references "
            "to specific articles, legal provisions or policy domains. "
            "Matching is pattern-based and applied on normalized text."
        ),
        "amend_help": (
            "Comma-separated verb fragments used to detect explicit amendment "
            "proposals (e.g. addition, deletion, modification). "
            "Matching is rule-based and applied on normalized text."
        ),
        "similarity_threshold_help": (
            "Minimum similarity percentage required for two comments "
            "to be grouped as duplicates in fuzzy mode. "
            "Higher values detect stricter template copies; "
            "lower values group more loosely similar variations."
        ),

        # --- Runtime ---
        "scraping": "Scraping and collecting comments...",
        "scraping_chapter": "Scraping chapter",
        "loaded_cache": "Results loaded from cache.",
        "completed": "Analysis completed.",
        "abort": "Abort",
        "no_comments": "No comments found or scraping aborted.",

        # --- Pipeline Steps ---
        "step_scrape": "Scraping chapters & comments",
        "step_normalize": "Normalizing text",
        "step_duplicates": "Detecting duplicates",
        "step_strict": "Calculating strict layer",
        "step_stats": "Computing statistics",
        "step_render": "Preparing outputs",

        # --- Duplicate Detection ---
        "duplicate_method_label": "Duplicate detection method",
        "duplicate_templates": "Duplicate Templates",
        "exact_match": "Exact match",
        "fuzzy_match": "Fuzzy match",
        "similarity_threshold": "Similarity threshold (%)",

        # --- Core Metrics ---
        "campaign": "Campaign Share (%)",
        "strict": "Strict Legislative Layer (%)",
        "mean": "Mean words",
        "median": "Median words",
        "std": "Std deviation",
        "max": "Max words",
        "total": "Total Comments",

        # --- Tooltips ---
        "campaign_help": (
            "Percentage of comments that are textual duplicates of other submissions. "
            "Approximates organized or template-based participation."
        ),
        "strict_desc": (
            "Percentage of comments containing both an article reference "
            "and an explicit amendment proposal."
        ),
        "mean_help": "Arithmetic average number of words per comment.",
        "median_help": "Middle value of word-count distribution.",
        "std_help": "Standard deviation of comment length.",
        "duplicate_templates_help": (
        "Number of distinct repeated comment texts (templates). "
        "Each template is counted once, regardless of how many times it appears."
        ),
        "targeted_layer_title": "Targeted Legislative Intervention Layer",
        "targeted_comments_detected": "targeted comments detected",
        "no_targeted": "No targeted legislative intervention comments detected.",
        "chapter": "Chapter",
        "article": "Article",
        "open_comment": "Open original comment",

        # --- Export buttons ---
        "export_comments_csv": "Export comments (CSV)",
        "export_metadata_json": "Export analysis metadata (JSON)",

        # --- Distribution ---
        "distribution": "Comment Length Distribution (Kernel Density Estimation)",
        "density": "Density",
        "mean_line": "Mean",
        "median_line": "Median",
        "word_count_label": "Word count",

        # --- Templates ---
        "top_templates": "Top Duplicate Templates",
        "occurrences": "Occurrences",
        "show_full_text": "Show full text",

        # --- Method Panel ---
        "method_panel": "Methodological Parameters",
        "execution_summary": "Execution Summary",
        "active_configuration": "Active Configuration",
        "timestamp": "Run timestamp",

        # --- Footer / Disclaimer ---
        "disclaimer_text": (
        "This application is a work in progress developed within the "
        "Postgraduate Programme «e-Government» of the University of the Aegean. "
        "It aims to extract and analyze comments submitted to public consultations "
        "on opengov.gr."
        ),
    
        "methodology_note": (
        "The analysis is fully rule-based and methodologically transparent. "
        "Metrics are calculated using predefined textual patterns and similarity thresholds, editable by users."
        ),
        "developed_by": "Developed with ❤️ by",

        "code_available": (
        "The source code is openly available at "
        "<a href='https://github.com/rinenweb/consultation-analyzer/' target='_blank'>GitHub</a>."
        ),

        "chapter_table_headers": [
            "Chapter ID",
            "Chapter Title",
            "Comment Count"
        ]
    },

    # ======================================================

    "el": {

        # --- Header ---
        "title": "Ανάλυση Δημόσιας Διαβούλευσης",
        "subtitle": "Εργαλείο διαφάνειας βασισμένο σε κανόνες.",

        # --- Input ---
        "input_label": "Εισάγετε URL διαβούλευσης από το opengov.gr",
        "run": "Εκτέλεση Ανάλυσης",

        # --- Advanced ---
        "advanced": "Προχωρημένες Ρυθμίσεις",
        "policy": "Λέξεις πολιτικής (διαχωρισμένες με κόμμα)",
        "amend": "Ρήματα τροποποίησης (διαχωρισμένα με κόμμα)",
        "policy_help": (
            "Λέξεις-κλειδιά (τμήματα λέξεων) διαχωρισμένες με κόμμα, "
            "που χρησιμοποιούνται για τον εντοπισμό αναφορών σε άρθρα, "
            "νομοθετικές διατάξεις ή θεματικά πεδία πολιτικής. "
            "Η ανίχνευση βασίζεται σε κανόνες και εφαρμόζεται στο κανονικοποιημένο κείμενο."
        ),
        "amend_help": (
            "Ρηματικά τμήματα διαχωρισμένα με κόμμα που χρησιμοποιούνται "
            "για τον εντοπισμό ρητών προτάσεων τροποποίησης "
            "(π.χ. προσθήκη, διαγραφή, αντικατάσταση). "
            "Η ανίχνευση βασίζεται σε κανόνες και εφαρμόζεται στο κανονικοποιημένο κείμενο."
        ),
        "similarity_threshold_help": (
            "Ελάχιστο ποσοστό ομοιότητας που απαιτείται ώστε δύο σχόλια "
            "να ομαδοποιηθούν ως διπλότυπα στη λειτουργία προσεγγιστικής ταύτισης. "
            "Υψηλότερες τιμές εντοπίζουν αυστηρότερες αντιγραφές template, "
            "ενώ χαμηλότερες τιμές ομαδοποιούν πιο χαλαρές παραλλαγές."
        ),

        # --- Runtime ---
        "scraping": "Εξαγωγή και ανάλυση σχολίων...",
        "scraping_chapter": "Εξαγωγή από κεφάλαιο",
        "loaded_cache": "Τα αποτελέσματα φορτώθηκαν από cache.",
        "completed": "Η ανάλυση ολοκληρώθηκε.",
        "abort": "Ακύρωση",
        "no_comments": "Δεν βρέθηκαν σχόλια ή η διαδικασία ακυρώθηκε.",

        # --- Βήματα Ανάλυσης ---
        "step_scrape": "Ανάκτηση κεφαλαίων & σχολίων",
        "step_normalize": "Κανονικοποίηση κειμένου",
        "step_duplicates": "Εντοπισμός διπλότυπων σχολίων",
        "step_strict": "Υπολογισμός strict layer",
        "step_stats": "Υπολογισμός στατιστικών",
        "step_render": "Προετοιμασία αποτελεσμάτων",

        # --- Duplicate Detection ---
        "duplicate_method_label": "Μέθοδος εντοπισμού διπλότυπων",
        "duplicate_templates": "Μοτίβα Διπλότυπων Σχολίων",
        "exact_match": "Ακριβής Ταύτιση",
        "fuzzy_match": "Προσεγγιστική Ταύτιση",
        "similarity_threshold": "Κατώφλι Ομοιότητας (%)",

        # --- Core Metrics ---
        "campaign": "Ποσοστό Campaign (%)",
        "strict": "Στρώμα Στοχευμένης Νομοθετικής Παρέμβασης (%)",
        "mean": "Μέσος Όρος Λέξεων",
        "median": "Διάμεσος",
        "std": "Τυπική Απόκλιση",
        "max": "Μέγιστος Αριθμός Λέξεων",
        "total": "Σύνολο Σχολίων",

        # --- Tooltips ---
        "campaign_help": (
            "Ποσοστό σχολίων που αποτελούν κειμενικά αντίγραφα άλλων υποβολών. "
            "Προσεγγίζει οργανωμένη συμμετοχή τύπου template."
        ),
        "strict_desc": (
            "Ποσοστό σχολίων που περιέχουν ταυτόχρονα αναφορά σε άρθρο "
            "και ρητή πρόταση τροποποίησης."
        ),
        "mean_help": "Αριθμητικός μέσος όρος λέξεων ανά σχόλιο.",
        "median_help": "Κεντρική τιμή της κατανομής μήκους.",
        "std_help": "Τυπική απόκλιση μήκους σχολίων.",
        # --- Tooltip ---
        "duplicate_templates_help": (
            "Αριθμός διαφορετικών επαναλαμβανόμενων κειμένων (templates). "
            "Κάθε template μετράται μία φορά, ανεξάρτητα από το πλήθος εμφανίσεων."
        ),
        "targeted_layer_title": "Στρώμα Στοχευμένης Νομοθετικής Παρέμβασης",
        "targeted_comments_detected": "στοχευμένα σχόλια εντοπίστηκαν",
        "no_targeted": "Δεν εντοπίστηκαν σχόλια στο στρώμα στοχευμένης παρέμβασης.",
        "chapter": "Κεφάλαιο",
        "article": "Άρθρο",
        "open_comment": "Άνοιγμα πρωτότυπου σχολίου",

        # --- Export buttons ---
        "export_comments_csv": "Εξαγωγή σχολίων (CSV)",
        "export_metadata_json": "Εξαγωγή μεταδεδομένων ανάλυσης (JSON)",

        # --- Distribution ---
        "distribution": "Κατανομή Μήκους Σχολίων (Εκτίμηση Πυκνότητας Kernel)",
        "density": "Πυκνότητα",
        "mean_line": "Μέσος Όρος",
        "median_line": "Διάμεσος",
        "word_count_label": "Αριθμός λέξεων",

        # --- Templates ---
        "top_templates": "Κορυφαία Επαναλαμβανόμενα Templates",
        "occurrences": "Εμφανίσεις",
        "show_full_text": "Εμφάνιση πλήρους κειμένου",

        # --- Method Panel ---
        "method_panel": "Μεθοδολογικές Παράμετροι",
        "execution_summary": "Σύνοψη Εκτέλεσης",
        "active_configuration": "Ενεργές Ρυθμίσεις",
        "timestamp": "Χρόνος Εκτέλεσης",

        # --- Footer / Disclaimer ---
        "disclaimer_text": (
            "Το παρόν εργαλείο αποτελεί έργο εν εξελίξει στο πλαίσιο του "
            "Μεταπτυχιακού Προγράμματος «Ηλεκτρονική Διακυβέρνηση» "
            "του Πανεπιστημίου Αιγαίου και αποσκοπεί στην εξαγωγή και "
            "ανάλυση σχολίων που υποβάλλονται σε δημόσιες διαβουλεύσεις "
            "στο opengov.gr."
        ),
        
        "methodology_note": (
            "Η ανάλυση είναι πλήρως βασισμένη σε κανόνες και μεθοδολογικά διαφανής. "
            "Οι μετρικές υπολογίζονται βάσει προκαθορισμένων γλωσσικών προτύπων "
            "και κατωφλιών ομοιότητας, τα οποία είναι διαθέσιμα προς επεξεργασία από τους χρήστες."
        ),
        
        "developed_by": "Developed with ❤️ by",

        "code_available": (
        "Ο πηγαίος κώδικας είναι δημόσια διαθέσιμος στο "
        "<a href='https://github.com/rinenweb/consultation-analyzer/' target='_blank'>GitHub</a>."
        ),

        "chapter_table_headers": [
            "ID Κεφαλαίου",
            "Τίτλος Κεφαλαίου",
            "Αριθμός Σχολίων"
        ]
    }
}
