"""
HTML templates for generating result pages
"""

def get_dcsbm_results_template(gica_report_path, gica_report_exists):
    """
    Returns HTML template for DCSBM results page
    
    Args:
        gica_report_path: Path to the GICA HTML report
        gica_report_exists: Boolean indicating if the report exists
        csv_files: List of CSV files in the output directory
        
    Returns:
        HTML content as a string
    """
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>DCSBM Analysis Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2c3e50; }
        h2 { color: #3498db; margin-top: 20px; }
        ul { list-style-type: none; padding-left: 10px; }
        li { margin: 10px 0; }
        a { color: #2980b9; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .report-section, .csv-section { 
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <h1>DCSBM Analysis Results</h1>
    
    <div class="report-section">
        <h2>GICA SBM Report</h2>
"""
    
    if gica_report_exists:
        html_content += f'        <p><a href="{gica_report_path}">View GICA SBM HTML Report</a></p>\n'
    else:
        html_content += '        <p>GICA SBM HTML Report not available</p>\n'
    
    html_content += """    </div>
</body>
</html>
"""
    return html_content


def get_aggregator_results_template(local_stats, global_stats):
    """
    Returns HTML template for aggregator results page with tables of local and global statistics
    
    Args:
        local_stats: Dictionary with site names as keys and lists of file paths as values
        global_stats: List of global statistics file paths
        
    Returns:
        HTML content as a string
    """
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>DCSBM Aggregator Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2c3e50; }
        h2 { color: #3498db; margin-top: 20px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; color: #333; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f1f1f1; }
        a { color: #2980b9; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .section { 
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <h1>DCSBM Aggregator Results</h1>
    
    <div class="section">
        <h2>Global Statistics</h2>
"""
    
    if global_stats:
        html_content += """        <table>
            <tr>
                <th>File Name</th>
                <th>Path</th>
            </tr>
"""
        for file_path in global_stats:
            file_name = file_path.split('/')[-1]
            html_content += f"""            <tr>
                <td>{file_name}</td>
                <td><a href="{file_path}">{file_path}</a></td>
            </tr>
"""
        html_content += "        </table>\n"
    else:
        html_content += "        <p>No global statistics available</p>\n"
    
    html_content += """    </div>
    
    <div class="section">
        <h2>Local Statistics by Site</h2>
"""
    
    if local_stats:
        for site, files in local_stats.items():
            html_content += f"""        <h3>Site: {site}</h3>
        <table>
            <tr>
                <th>File Name</th>
                <th>Path</th>
            </tr>
"""
            for file_path in files:
                file_name = file_path
                html_content += f"""            <tr>
                <td>{file_name}</td>
                <td>{file_path}</p></td>
            </tr>
"""
            html_content += "        </table>\n"
    else:
        html_content += "        <p>No local statistics available</p>\n"
    
    html_content += """    </div>
</body>
</html>
"""
    return html_content