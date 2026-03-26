/**
 * Shogi Board — Interactive Component (Streamlit v2 API)
 *
 * Ported from the v1 iframe-based component. Key differences:
 * - No iframe, no postMessage — runs in the page DOM
 * - Uses component.data instead of render event args
 * - Uses setTriggerValue/setStateValue instead of setComponentValue
 * - parentElement is the mount point (shadow root or div)
 */
export default function(component) {
  var CELL_SIZE = 48;
  var parentEl = component.parentElement;
  var root = parentEl.querySelector("#board-root") || parentEl;

  // State
  var focusRow = 0;
  var focusCol = 0;
  var selectedRow = null;
  var selectedCol = null;
  var boardFocused = false;
  var blurTimeout = null;

  // Initialize from data
  var d = component.data || {};
  if (d.selected_square && typeof d.selected_square === "object") {
    focusRow = d.selected_square.row || 0;
    focusCol = d.selected_square.col || 0;
    selectedRow = focusRow;
    selectedCol = focusCol;
  }

  // =========================================================================
  // Heatmap normalization (ported from _compute_heatmap_overlay in Python)
  // =========================================================================
  function computeHeatmapOverlay(heatmap) {
    if (!heatmap) return null;
    var EPSILON = 1e-10;
    var flat = [];
    for (var r = 0; r < 9; r++) {
      for (var c = 0; c < 9; c++) {
        var v = heatmap[r][c];
        if (v > EPSILON) flat.push(v);
      }
    }
    if (flat.length === 0) return null;
    var logMin = Math.log(Math.min.apply(null, flat) + EPSILON);
    var logMax = Math.log(Math.max.apply(null, flat) + EPSILON);
    var logRange = (logMax > logMin) ? (logMax - logMin) : 1.0;
    var result = [];
    for (var r2 = 0; r2 < 9; r2++) {
      var row = [];
      for (var c2 = 0; c2 < 9; c2++) {
        var v2 = heatmap[r2][c2];
        if (v2 <= EPSILON) { row.push(0.0); }
        else { row.push(Math.max(0.0, Math.min(1.0, (Math.log(v2 + EPSILON) - logMin) / logRange))); }
      }
      result.push(row);
    }
    return result;
  }

  // =========================================================================
  // Sanitization
  // =========================================================================
  function escHtml(s) {
    // Escape HTML special characters to prevent injection
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function isSafeDataUri(uri) {
    // Only allow data: URIs (our SVG piece images) and reject anything else
    return typeof uri === "string" && uri.indexOf("data:") === 0;
  }

  // =========================================================================
  // Piece helpers
  // =========================================================================
  function pieceAriaLabel(piece) {
    if (!piece) return "empty square";
    return escHtml(piece.color) + " " + escHtml(piece.type.replace(/_/g, " "));
  }

  function pieceImageKey(piece) {
    var PROMOTED_TYPES = [
      "promoted_pawn", "promoted_lance", "promoted_knight",
      "promoted_silver", "promoted_bishop", "promoted_rook"
    ];
    var ptype = piece.type;
    if (piece.promoted && PROMOTED_TYPES.indexOf(ptype) === -1) {
      ptype = "promoted_" + ptype;
    }
    return ptype + "_" + piece.color;
  }

  // =========================================================================
  // Board rendering
  // =========================================================================
  function renderBoard() {
    var boardState = d.board_state || {};
    var heatmap = d.heatmap || null;
    var pieceImages = d.piece_images || {};
    var board = boardState.board || [];
    var moveCount = boardState.move_count || 0;
    var currentPlayer = boardState.current_player || "unknown";

    if (board.length === 0) {
      if (!root.querySelector("#shogi-board")) {
        root.innerHTML = '<p style="text-align:center;color:#888;">No board data</p>';
      }
      return;
    }

    var overlay = computeHeatmapOverlay(heatmap);
    var html = "";

    // Column headers
    html += '<thead role="rowgroup"><tr>';
    html += '<th scope="col" aria-hidden="true"></th>';
    for (var f = 9; f >= 1; f--) { html += '<th scope="col">' + f + '</th>'; }
    html += '</tr></thead>';

    // Board rows
    html += '<tbody role="rowgroup">';
    for (var r = 0; r < board.length; r++) {
      var rank = r + 1;
      var row = board[r];
      html += '<tr><th scope="row">' + rank + '</th>';

      for (var c = 0; c < 9; c++) {
        var piece = row[c];
        var fileNum = 9 - c;
        var aria = pieceAriaLabel(piece);
        var cellLabel = "File " + fileNum + " Rank " + rank + ": " + aria;
        var bg = ((r + c) % 2 === 0) ? "#f5deb3" : "#deb887";

        // Promotion zone tint
        var zoneTint = "";
        if (rank <= 3) {
          zoneTint = "background-image:linear-gradient(rgba(100,149,237,0.08),rgba(100,149,237,0.08));";
        } else if (rank >= 7) {
          zoneTint = "background-image:linear-gradient(rgba(220,80,80,0.08),rgba(220,80,80,0.08));";
        }

        // Rank 3/6 thicker border
        var borderBottom = "1px solid #8b7355";
        if (rank === 3 || rank === 6) { borderBottom = "2.5px solid #6b5335"; }

        // Heatmap: solid background replacement, white-to-navy scale.
        // No transparency — directly overrides the cell background colour.
        // Subtle texture preserves the grid pattern: diagonal lines on
        // "dark" squares, no texture on "light" squares.
        var heatStyle = "";
        if (overlay && overlay[r][c] > 0.01) {
          var h = overlay[r][c];  // 0..1
          var hr = Math.round(255 * (1 - h));
          var hg = Math.round(255 * (1 - h * 0.9));
          var hb = Math.round(255 * (1 - h * 0.7));
          bg = "rgb(" + hr + "," + hg + "," + hb + ")";
          // "Dark" squares get fine diagonal lines to maintain grid feel
          if ((r + c) % 2 !== 0) {
            zoneTint = "background-image:repeating-linear-gradient(" +
              "135deg,rgba(0,0,0,0.06),rgba(0,0,0,0.06) 1px,transparent 1px,transparent 4px);";
          }
        }

        var tabIdx = (r === focusRow && c === focusCol) ? "0" : "-1";
        var cls = (r === selectedRow && c === selectedCol) ? ' class="selected"' : "";
        var isSelected = (r === selectedRow && c === selectedCol);

        // Cell content
        var content = "";
        if (piece) {
          var key = pieceImageKey(piece);
          var svgUri = pieceImages[key] || "";
          if (svgUri && isSafeDataUri(svgUri)) {
            var imgSz = CELL_SIZE - 8;
            content = '<img src="' + escHtml(svgUri) + '" width="' + imgSz +
              '" height="' + imgSz + '" alt="" style="pointer-events:none;">';
          } else {
            var label = escHtml(piece.type.charAt(0).toUpperCase());
            if (piece.promoted) label = "+" + label;
            var txtColor = (piece.color === "black") ? "#000" : "#8b0000";
            content = '<span style="color:' + txtColor + ';font-weight:bold;pointer-events:none;">' + label + '</span>';
          }
        }

        // Non-colour cue for heatmap: corner dot scaled by heat intensity.
        // Ensures heatmap is readable for red-green colorblind users.
        // Uses inline absolute positioning within the position:relative cell.
        var heatDot = "";
        if (overlay && overlay[r][c] > 0.01) {
          var dotSize = Math.max(4, Math.round(3 + overlay[r][c] * 8));  // 4-11px
          heatDot = '<span style="position:absolute;top:1px;right:1px;width:' + dotSize +
            'px;height:' + dotSize + 'px;border-radius:50%;background:#004d33;' +
            'pointer-events:none;z-index:3;" aria-hidden="true"></span>';
        }

        html += '<td role="gridcell" tabindex="' + tabIdx + '"' +
          ' data-row="' + r + '" data-col="' + c + '"' +
          ' aria-label="' + cellLabel + '"' +
          ' aria-selected="' + (isSelected ? "true" : "false") + '"' +
          ' aria-rowindex="' + (r + 1) + '" aria-colindex="' + (c + 1) + '"' +
          cls +
          ' style="width:' + CELL_SIZE + 'px;height:' + CELL_SIZE + 'px;' +
          'background:' + bg + ';text-align:center;vertical-align:middle;' +
          'border:1px solid #8b7355;border-bottom:' + borderBottom + ';cursor:pointer;position:relative;' +
          zoneTint + heatStyle + '">' + content + heatDot + '</td>';
      }
      html += '</tr>';
    }
    html += '</tbody>';

    var gridLabel = "Shogi board position, move " + moveCount + ", " +
      currentPlayer.charAt(0).toUpperCase() + currentPlayer.slice(1) +
      " to play. Black (Sente) plays from bottom, White (Gote) from top.";

    root.innerHTML = '<table id="shogi-board" role="grid" ' +
      'aria-rowcount="9" aria-colcount="9" ' +
      'style="border-collapse:collapse;margin:auto;" ' +
      'aria-label="' + gridLabel + '">' + html + '</table>';

    if (boardFocused) {
      var focusCell = getCell(focusRow, focusCol);
      if (focusCell) focusCell.focus();
    }
  }

  // =========================================================================
  // Helpers
  // =========================================================================
  function getCell(row, col) {
    return root.querySelector('td[data-row="' + row + '"][data-col="' + col + '"]');
  }

  function moveFocusTo(newRow, newCol) {
    newRow = Math.max(0, Math.min(8, newRow));
    newCol = Math.max(0, Math.min(8, newCol));
    if (newRow === focusRow && newCol === focusCol) return;
    var oldCell = getCell(focusRow, focusCol);
    var newCell = getCell(newRow, newCol);
    if (oldCell) oldCell.setAttribute("tabindex", "-1");
    if (newCell) { newCell.setAttribute("tabindex", "0"); newCell.focus(); }
    focusRow = newRow;
    focusCol = newCol;
  }

  function toggleSelect(row, col) {
    if (selectedRow === row && selectedCol === col) {
      selectedRow = null;
      selectedCol = null;
      component.setTriggerValue("selection", {type: "deselect"});
    } else {
      selectedRow = row;
      selectedCol = col;
      component.setTriggerValue("selection", {row: row, col: col, type: "select"});
    }
    renderBoard();
  }

  // =========================================================================
  // Event handlers (delegation on root)
  // =========================================================================
  root.addEventListener("click", function(e) {
    var cell = e.target.closest("td[role='gridcell']");
    if (!cell) return;
    var row = parseInt(cell.dataset.row);
    var col = parseInt(cell.dataset.col);
    moveFocusTo(row, col);
    toggleSelect(row, col);
  });

  root.addEventListener("keydown", function(e) {
    var cell = e.target.closest("td[role='gridcell']");
    if (!cell) return;
    var row = parseInt(cell.dataset.row);
    var col = parseInt(cell.dataset.col);

    switch (e.key) {
      case "ArrowUp":    e.preventDefault(); moveFocusTo(row - 1, col); break;
      case "ArrowDown":  e.preventDefault(); moveFocusTo(row + 1, col); break;
      case "ArrowLeft":  e.preventDefault(); moveFocusTo(row, col - 1); break;
      case "ArrowRight": e.preventDefault(); moveFocusTo(row, col + 1); break;
      case "Home":
        e.preventDefault();
        moveFocusTo(e.ctrlKey ? 0 : row, 0);
        break;
      case "End":
        e.preventDefault();
        moveFocusTo(e.ctrlKey ? 8 : row, 8);
        break;
      case "PageUp":  e.preventDefault(); moveFocusTo(0, col); break;
      case "PageDown": e.preventDefault(); moveFocusTo(8, col); break;
      case "Enter":
      case " ":
        e.preventDefault();
        toggleSelect(focusRow, focusCol);
        break;
      case "Escape":
        e.preventDefault();
        if (selectedRow !== null) {
          selectedRow = null;
          selectedCol = null;
          component.setTriggerValue("selection", {type: "deselect"});
          renderBoard();
        }
        break;
    }
  });

  // Focus tracking
  root.addEventListener("focusin", function(e) {
    if (e.target.matches && e.target.matches("td[role='gridcell']")) {
      clearTimeout(blurTimeout);
      if (!boardFocused) {
        boardFocused = true;
        component.setStateValue("board_focused", true);
      }
    }
  });

  root.addEventListener("focusout", function() {
    clearTimeout(blurTimeout);
    blurTimeout = setTimeout(function() {
      var active = document.activeElement;
      if (!active || !active.matches || !active.matches("td[role='gridcell']")) {
        boardFocused = false;
        component.setStateValue("board_focused", false);
      }
    }, 100);
  });

  // Initial render
  renderBoard();
}
